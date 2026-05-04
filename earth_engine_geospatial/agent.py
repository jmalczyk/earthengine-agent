"""An Earth Engine enabled agent."""

import functools
import logging
import os
import dotenv

dotenv.load_dotenv(override=True)

import ee
import google
from google.adk.agents import llm_agent

from . import prompt, tools
from google.cloud import aiplatform

env_values = dotenv.dotenv_values(".env")
_PROJECT_ID = env_values.get("GOOGLE_CLOUD_PROJECT")

if not _PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT must be set in .env file.")
if "cloudshell" in _PROJECT_ID:
    raise ValueError("Detected cloudshell in project ID. Please set a valid project ID in .env.")

_LOCATION = "us-central1"


@functools.cache
def _initialize_earth_engine():
    """Initializes the Earth Engine client exactly once."""
    try:
        if not _PROJECT_ID:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable not set."
            )
        else:
            print(f"GOOGLE_CLOUD_PROJECT is {_PROJECT_ID}")
            print(f"GOOGLE_CLOUD_LOCATION is {_LOCATION}")
        
        scopes = [
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
        credentials, _ = google.auth.default(scopes=scopes)

        ee.Initialize(
            credentials,
            project=_PROJECT_ID,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )
        logging.info(
            "Earth Engine initialized successfully for project: %s", _PROJECT_ID
        )
        
        # Initialize aiplatform with the project ID and location
        aiplatform.init(project=_PROJECT_ID, location=_LOCATION)
        logging.info("Vertex AI initialized successfully for project: %s in region: %s", _PROJECT_ID, _LOCATION)

    except Exception as e:
        logging.exception("Failed to initialize Earth Engine: %s", e)
        raise


_initialize_earth_engine()

root_agent = llm_agent.Agent(
    name="ee_agent",
    model="gemini-3-flash-preview",
    description="Agent to answer geo questions using Google Earth Engine.",
    tools=[
        tools.get_2017_2025_annual_changes,
        tools.generate_geojson_for_location,
        tools.generate_change_map_image,
        tools.create_interactive_map,
        tools.get_dynamic_world_landcover_areas,
    ],

    instruction=prompt.root_agent_prompt,
)
