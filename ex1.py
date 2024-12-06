"""
Author: Scriptone
Created: 2024/10/15
Description: This module exports BlueprintController for managing blueprints in the database.
"""

from app.controllers.diskOfferings import DiskOfferingController
from app.controllers.serviceOfferings import ServiceOfferingController
from app.controllers.templates import TemplateController
from app.db import (
    BlueprintNetworkVirtualMachine,
    Blueprints,
    BlueprintsNetworks,
    BlueprintsVirtualMachines,
    Courses,
)
from app.models.blueprint import (
    BlueprintResponse,
    CreateBlueprintModel,
    NetworksResponse,
)
from app.utils.constants import *
from app.utils.validation import isIPInCIDR, isValidCIDR, isValidIP
from fastapi import HTTPException
from fastapi.logger import logger
from sqlalchemy.orm import Session, joinedload

# Context: Blueprints are designs that in their simplest form represent a virtual machine connected to a public or private network (or multiple VMs and networks)


# Each private network is placed inside a Virtual Private Cloud (VPC) that has a CIDR address
# We need to validate that the gateway IP is within the VPC CIDR and the netmask is valid, this way we can provide meaningful error messages to the user.
# Because internal server errors that rise from continueing the process wont be meaningful to the user.
def validateNetwork(network: NetworksResponse, vpc_cidr: str):
    type = network.network_type
    if type not in [
        "public",
        "private",
    ]:  # We only allow these 2 types, anything else would be considered invalid
        raise ValueError(f"Invalid network type: {type}")

    # The following code is only for private networks
    # It'll basically validate that the gateway and netmask are valid IPs and within the VPC CIDR and otherwise throw a meaningful error message so the user can correct them.
    gateway = network.gateway if type == "private" else None
    if gateway and not isValidIP(gateway):
        raise ValueError(f"Invalid gateway IP: {gateway}")
    if gateway and not isIPInCIDR(gateway, vpc_cidr):
        raise ValueError(f"Gateway IP {gateway} not in VPC CIDR {vpc_cidr}")
    netmask = network.netmask if type == "private" else None
    if netmask and not isValidIP(netmask):
        raise ValueError(f"Invalid netmask IP: {netmask}")
    return type, gateway, netmask


# Controller to perform all CRUD operations on blueprints, note that I am not using any try/except statements because I am using a global exception handler for simplicity. (ex2)
class BlueprintController:
    @staticmethod
    async def get_blueprints(
        db: Session, user_info: dict
    ):  # db and userinfo are passed as arguments from the router, which got injected by fastAPI Dependency Injection

        # There are 3 roles so far, sysadmins, tutors and students.
        # Students should only be able to see their own blueprint, hence the onlyVisible flag is used here.
        roles = user_info.get("roles", [])
        onlyVisible = "sysadmin" not in roles and "tutor" not in roles

        builder = db.query(
            Blueprints
        )  # SQLAlchemy is used here because I like it over raw SQL.

        if onlyVisible:
            builder = builder.filter_by(
                visible=True
            )  # Blueprints contain a visible property that indicate wether students can see it or not.

        # We're using mysql here, I have 4 tables (blueprints, networks, vms and a table to connect vms with networks), we need to perform a JOIN operation to fetch all related data.
        blueprints = builder.options(
            joinedload(Blueprints.networks),
            joinedload(Blueprints.vms).joinedload(BlueprintsVirtualMachines.networks),
        ).all()

        return blueprints

    @staticmethod
    async def get_blueprint(db: Session, blueprint_id: int) -> BlueprintResponse:
        # Basically the same as the above so I won't be going, I could've refactored here so both functions get the same query builder but I like the readability of this one.
        blueprint = (
            db.query(Blueprints)
            .options(
                joinedload(Blueprints.networks),
                joinedload(Blueprints.vms).joinedload(
                    BlueprintsVirtualMachines.networks
                ),
            )
            .filter_by(id=blueprint_id)
            .first()
        )

        # Since the above query can return None if the blueprint doesn't exist, we need to check that and inform the user appropriately with a 404 error.
        if not blueprint:
            raise HTTPException(status_code=404, detail="Blueprint not found")
        return blueprint

    @staticmethod
    def _validate_course(db: Session, course_id: int):
        # At the time of writing I realised this should be in the CourseController but now you have an example of a different class being used
        # Same as the above, find the course if it exists, otherwise raise a 404 error.
        course = db.query(Courses).filter_by(id=course_id).first()
        if not course and course_id:
            raise HTTPException(status_code=404, detail="Course not found")
        return course

    # Check if the CIDR of the vpc is valid, inform the user if not. FYI ValueErrors are also catched globally and return the appropriate 422.
    @staticmethod
    def _validate_vpc_cidr(vpc_cidr: str):
        if not isValidCIDR(vpc_cidr):
            raise ValueError(f"Invalid VPC CIDR: {vpc_cidr}")

    @staticmethod
    def _set_blueprint_fields(blueprint_obj, blueprint_data):
        # Um, just setting the fields based on the provided data.
        blueprint_obj.name = blueprint_data.name
        blueprint_obj.visible = blueprint_data.visible
        blueprint_obj.description = (
            blueprint_data.description or blueprint_obj.description
        )
        blueprint_obj.allow_sharing = blueprint_data.allow_sharing
        blueprint_obj.image = blueprint_data.image
        blueprint_obj.course_id = (
            int(blueprint_data.course_id)
            if blueprint_data.course_id
            else blueprint_obj.course_id
        )
        blueprint_obj.lease_time = blueprint_data.lease_time
        blueprint_obj.max_snapshots = blueprint_data.max_snapshots
        blueprint_obj.vpc_cidr = blueprint_data.vpc_cidr
        blueprint_obj.draft = False if blueprint_obj.visible else blueprint_data.draft

    @staticmethod
    def _create_networks(db: Session, blueprint_id: int, networks, vpc_cidr: str):
        # Create a new network for each network in the blueprint, and then associate them with the blueprint.
        blueprint_networks = []
        for network in networks:
            # Validate the network and create a new network in the database.
            type, gateway, netmask = validateNetwork(network, vpc_cidr)
            new_network = BlueprintsNetworks(
                blueprint_id=blueprint_id,
                network_type=type,
                name=network.name,
                dhcp=network.dhcp,
                gateway=gateway,
                netmask=netmask,
                description=network.description,
                meta=network.meta,
            )
            blueprint_networks.append(new_network)
            db.add(new_network)
        db.flush()
        # Flushing the database to make sure all the new networks are "saved" -> we can get the ids now.
        return blueprint_networks

    @staticmethod
    async def _create_vms(db: Session, blueprint_id: int, vms, blueprint_networks):
        for vm in vms:
            await BlueprintController._create_vm(
                db, vm, blueprint_id, blueprint_networks
            )

    @staticmethod
    async def _create_vm(db: Session, vm, blueprint_id: int, blueprint_networks):
        # Doing some validation to make sure the provided values are legit, did the user select a valid template, if not -> 422
        templateid = (await TemplateController.get_template_by_id(vm.templateid))["id"]
        if not templateid:
            raise ValueError(f"Template not found for ID {vm.templateid}")

        # Users can't select specific service or disk offerings, this would mean I'd have to make them all myself.
        # Instead we check if they exists and otherwise create them.
        serviceOffering = await get_or_create_service_offering(vm.cpu, vm.ram)
        if not serviceOffering:
            raise ValueError(
                f"Service offering not found for CPU {vm.cpu} and RAM {vm.ram}"
            )
        diskOffering = await get_or_create_disk_offering(vm.disk)
        if not diskOffering:
            raise ValueError(f"Disk offering not found for size {vm.disk}")

        # Create the vm (yes I'm explaining the what!!)
        new_vm = BlueprintsVirtualMachines(
            blueprint_id=blueprint_id,
            diskofferingid=diskOffering,
            templateid=templateid,
            serviceofferingid=serviceOffering,
            username=vm.user.username,
            password=vm.user.password,
            displayname=vm.displayname,
            userdata=vm.userdata,
            meta=vm.meta,
        )
        db.add(new_vm)
        db.flush()
        # Flush for id, needed to connect the vm with the networks.
        BlueprintController._create_interfaces(
            db, vm.networks, new_vm.id, blueprint_networks
        )

    @staticmethod
    def _create_interfaces(db: Session, interfaces, vm_id: int, blueprint_networks):
        # You know the drill, validate and connect the interfaces to the networks.
        for interface in interfaces:

            # Basically we have no "sexy" way to connect a vm to a network when we send it over. A VM contains a field 'networks' which has a meta field
            # that looks like "23428383242342#10#20" where the first part is the id of the network,
            # the second part is the position of the node (see canvas on live demo where a blueprint is briefly created) which is irrelevant for this example.
            # Now that we have the id of the interface and compare it with the networks.
            network = next(
                (
                    net
                    for net in blueprint_networks
                    if net.meta.startswith(interface.meta.split("#")[0])
                ),
                None,
            )
            if network is None:
                raise ValueError("Network not found for interface")
            ip = interface.ipaddress if network.network_type == "private" else None
            if ip and not isValidIP(ip):
                raise ValueError(f"Invalid IP address: {ip}")
            if ip and not isIPInCIDR(ip, network.gateway):
                raise ValueError(
                    f"IP address {ip} not in network CIDR {network.gateway}"
                )
            new_interface = BlueprintNetworkVirtualMachine(
                virtual_machine_id=vm_id,
                network_id=network.id,
                ipaddress=ip,
            )
            db.add(new_interface)

    @staticmethod
    async def create_blueprint(db: Session, blueprint: CreateBlueprintModel):

        # FYI, fastAPI rollbacks automatically when an error occurs.

        # Now we have fun and call all the nice functions.
        BlueprintController._validate_course(db, blueprint.course_id)
        BlueprintController._validate_vpc_cidr(blueprint.vpc_cidr)
        new_blueprint = Blueprints(domain_id=default_domain_id)
        BlueprintController._set_blueprint_fields(new_blueprint, blueprint)
        db.add(new_blueprint)
        db.flush()
        blueprint_networks = BlueprintController._create_networks(
            db, new_blueprint.id, blueprint.networks, blueprint.vpc_cidr
        )
        await BlueprintController._create_vms(
            db, new_blueprint.id, blueprint.vms, blueprint_networks
        )
        db.commit()
        return new_blueprint

    # I'm pretty sure I have sufficient comments as this only had to be 50 lines, enjoy the remaining lines. Most of the lines were previously explained though.
    @staticmethod
    async def update_blueprint(
        db: Session, blueprint_id: int, blueprint: CreateBlueprintModel
    ):
        try:
            BlueprintController._validate_course(db, blueprint.course_id)
            existing_blueprint = db.query(Blueprints).filter_by(id=blueprint_id).first()
            if not existing_blueprint:
                logger.error(f"Blueprint ID {blueprint_id} not found")
                raise HTTPException(status_code=404, detail="Blueprint not found")
            BlueprintController._validate_vpc_cidr(blueprint.vpc_cidr)
            BlueprintController._set_blueprint_fields(existing_blueprint, blueprint)
            db.query(BlueprintsNetworks).filter_by(blueprint_id=blueprint_id).delete()
            db.query(BlueprintsVirtualMachines).filter_by(
                blueprint_id=blueprint_id
            ).delete()
            db.flush()
            blueprint_networks = BlueprintController._create_networks(
                db, blueprint_id, blueprint.networks, blueprint.vpc_cidr
            )
            await BlueprintController._create_vms(
                db, blueprint_id, blueprint.vms, blueprint_networks
            )
            db.commit()
            return existing_blueprint
        except Exception as e:
            raise e

    @staticmethod
    async def patch_blueprint(
        db: Session, blueprint_id: int, blueprint_data: CreateBlueprintModel
    ):
        try:
            course_id = blueprint_data.get("course_id")  # Or the appropriate key
            course = db.query(Courses).filter_by(id=course_id).first()
            if not course and course_id:
                raise HTTPException(status_code=404, detail="Course not found")
            # Retrieve existing blueprint
            blueprint = db.query(Blueprints).filter_by(id=blueprint_id).first()
            if not blueprint:
                logger.info(f"Blueprint ID {blueprint_id} not found")
                raise HTTPException(status_code=404, detail="Blueprint not found")

            # Update blueprint fields
            blueprint.name = blueprint_data.get("name", blueprint.name)
            blueprint.visible = blueprint_data.get("visible", blueprint.visible)
            blueprint.allow_sharing = blueprint_data.get(
                "allow_sharing", blueprint.allow_sharing
            )
            blueprint.description = blueprint_data.get(
                "description", blueprint.description
            )
            blueprint.image = blueprint_data.get("image", blueprint.image)
            blueprint.course_id = int(course_id) if course_id else blueprint.course_id
            blueprint.lease_time = blueprint_data.get(
                "lease_time", blueprint.lease_time
            )
            blueprint.max_snapshots = blueprint_data.get(
                "max_snapshots", blueprint.max_snapshots
            )
            vpc_cidr = blueprint_data.get("vpc_cidr", blueprint.vpc_cidr)
            if not isValidCIDR(vpc_cidr):
                raise ValueError(f"Invalid VPC CIDR: {vpc_cidr}")
            blueprint.vpc_cidr = vpc_cidr
            blueprint.draft = (
                False
                if blueprint.visible
                else blueprint_data.get("draft", blueprint.draft)
            )

            db.commit()
        except Exception as e:
            raise e

    @staticmethod
    async def delete_blueprint(db: Session, blueprint_id: int):
        db.query(Blueprints).filter_by(id=blueprint_id).delete()
        db.commit()


async def get_or_create_service_offering(cpu: int, ram: int):
    params = {"cpunumber": cpu, "memory": ram}
    result = await ServiceOfferingController.get_service_offerings(params)

    if not result or len(result) == 0:
        params.update(
            {
                "cpuspeed": 1000,
                "name": f"{cpu}-{ram}",
            }
        )
        result = await ServiceOfferingController.create_service_offering(params)
        serviceOffering = result[0]["id"]
    else:
        serviceOffering = result[0]["id"]
    return serviceOffering


async def get_or_create_disk_offering(disk_size: int):
    params = {"disksize": disk_size}
    result = await DiskOfferingController.get_disk_offerings(params)
    if not result or len(result) == 0:
        params.update(
            {
                "name": str(disk_size),
                "disksize": disk_size,
            }
        )
        result = await DiskOfferingController.create_disk_offering(params)
        diskOffering = result["id"]
    else:
        diskOffering = result[0]["id"]
    return diskOffering
