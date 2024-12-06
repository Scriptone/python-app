"""
Author: Scriptone
Created: 2024/10/15
Description: FastAPI application setup for the StudentCloud API.
"""

import os

import requests
from app.classes.CloudstackClient import CloudstackClient
from app.classes.TokenValidator import TokenValidator
from app.classes.WebSocket import WebSocketHandler
from app.routes import api_router
from app.utils.custom_logger import LoggingMiddleware
from app.utils.logger import logger, logstash_logger
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy.exc import SQLAlchemyError

app = FastAPI(title="StudentCloud API", version="1.0")

# Add logstash middleware for logging user actions (I love this so much)
app.add_middleware(LoggingMiddleware, logger=logstash_logger)

# Add cors middleware, 2024... FastAPI handles most of it, I just allow my localhost, dev, staging and production servers.
cors_origins = os.getenv("CORS_ORIGIN_WHITELIST", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Using self signed certificates for now, now we can use https "properly"
self_signed_cert_path = os.path.join(os.path.dirname(__file__), "self-signed-cert.pem")
trusted_session = requests.Session()
trusted_session.verify = self_signed_cert_path

# Init the cloudstack client, this is a wrapper class that allows me to make API calls to the Apache CloudStack API (https://cloudstack.apache.org/)
CloudstackClient(
    api=os.getenv("CLOUDSTACK_API_URL"),
    apikey=os.getenv("API_KEY"),
    secret=os.getenv("SECRET_KEY"),
    session=trusted_session,
)


# Some API calls to cloudstack are asynchronous and if we would completely wait for the request to complete, we would block the event loop. This is why we
# can initiate the job (with cloudstack) and then let the rest happen in a background task. The solution to notify the user when the job is done is through a WebSocket connection.
websocket_handler = WebSocketHandler()


# Expose an endpoint for WebSocket connections. When a client connects, we send an ID to them so we can use it in the REST endpoints.
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # This connects them and sends the id.
    await websocket_handler.connect(websocket)

    # This keeps our connection alive, we also handle the subscriptions here.
    try:
        while True:
            message = await websocket.receive_json()

            topic = message.get("topic")
            data = message.get("data")

            logger.info(f"Received message: {message}, topic: {topic}, data: {data}")
            if topic == "subscribe":
                await websocket_handler.subscribe(websocket, data)
            elif topic == "unsubscribe":
                await websocket_handler.unsubscribe(websocket, data)
            elif topic == "message":
                await websocket_handler.send_message(data, topic=topic)
    except WebSocketDisconnect:
        await websocket_handler.disconnect(websocket)


# The frontend uses keycloak for authentication, keycloak gives us a token that I included in the requests to this API, we decode it to get the user info.
token_validator = TokenValidator(secret_key="12345", algorithm="HS256")

app.include_router(api_router)


# These are the global exception handles as mentioned in example 1, server errors shouldn't happen and return a 500.
@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.info(f"SQLAlchemy error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Database error"},
    )


# The next 2 are both validation error handlers, e.g. 404 error because it couldn't find the blueprint.
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# I have many ValueErrors and raising 422 myself would add quite some line.
@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    logger.info(f"Value error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


# Technically this block shouldn't run ever, but it's good practice to always handle exceptions.
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.info(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Middleware to validate the token. This is used to ensure that only authenticated users can access the API. (Or if you're in development mode)
@app.middleware("http")
async def token_middleware(request: Request, call_next):

    token = request.headers.get("x-access-token")
    development = os.getenv("FLASK_ENV") == "development"

    # Some endpoints don't require a token, for now I only added the swagger docs.
    exempt_endpoints = ["docs", "restx_doc.static", "specs", "metrics"]

    # Also skip for options.
    if request.url.path in exempt_endpoints or request.method == "OPTIONS":
        response = await call_next(request)
    else:
        if not token and not development:
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED, content="Token is required"
            )
        try:
            user_info = token_validator.validate_token(token, development)
            request.state.user = user_info
        except HTTPException as e:
            return Response(status_code=e.status_code, content=e.detail)

        response = await call_next(request)

    return response
