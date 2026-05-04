# Earth Engine Geospatial Agent

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://console.cloud.google.com/cloudshell/editor?cloudshell_tutorial=README.md&ephemeral=false&cloudshell_git_repo=https://github.com/jmalczyk/earthengine-agent.git)

## Overview

This directory contains a [Google Earth Engine](https://earthengine.google.com/)
enabled ADK agent implemented as a simple chatbot.

The agent has tools to:
- Compute the area of annual change between 2017-2025 in a geometry provided by the user, using the [AlphaEarth Satellite Embeddings dataset](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL).
- Generate an Earth Engine XYZ tile URL pattern detailing the magnitude of change.
- Create an interactive map HTML file with layer controls (opacity and visibility) and render it in the ADK output.

The tools convert the GeoJSON to an `ee.Geometry` and send it to the Earth Engine server for processing. The agent uses the computation results and the coordinates in the geometry to reason about land cover change and provide an interactive visualization.

## Agent Details

| Feature | Description |
| --- | --- |
| **Interaction Type** | Conversational |
| **Complexity** | Easy |
| **Agent Type** | Single Agent |
| **Components** | Tools: Earth Engine, Interactive Map |
| **Vertical** | Geospatial |

## Setup and Installation

1. **Prerequisites**
   * Python 3.10+
   * [uv](https://docs.astral.sh/uv/) for dependency management
   * A Google Cloud project with the Earth Engine API, Vertex AI API, and Google Maps Platform APIs enabled (**Geocoding API** and **Maps JavaScript API**).
   * [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)

2. **Installation**
   ```bash
   git clone https://github.com/google/adk-samples.git
   cd adk-samples/python/agents/earth-engine-geospatial
   uv sync
   ```

3. **Configuration**
   Copy `.env.example` to `.env` and fill in the required values. Ensure that automated tools like Gemini-CLI generate the correct values for all configurations and do NOT rename variables:

   ```bash
   cp .env.example .env
   ```

   Fill in the following variables in `.env`:
   * `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID.
   * `GOOGLE_MAPS_API_KEY`: Your Google Maps API key (with Geocoding and Maps JavaScript APIs enabled).


   Authenticate:

   ```bash
   gcloud auth application-default login
   ```
   You may also need to run `earthengine authenticate`.

## Running in Google Cloud Shell

If you are running this agent in Google Cloud Shell, you can use the following instructions to set up your environment.

### Quick Start (Automated Setup)

You can download and run the startup script in a single command in Cloud Shell:

```bash
curl -s -O https://raw.githubusercontent.com/jmalczyk/earthengine-agent/main/startup.sh && chmod +x startup.sh && ./startup.sh
```

The script will automatically clone the repository if you are not already inside it.

Alternatively, if you have already cloned the repository:

```bash
./startup.sh
```

*Note: You will still need to authenticate (see step 4 below).*

### Manual Setup

These instructions are designed to be easily followed by automated assistants like Gemini-CLI.

### 1. Enable Necessary APIs

Run the following command to enable all required APIs for this agent (Earth Engine, Vertex AI, Geocoding API, and Maps JavaScript API):

```bash
gcloud services enable earthengine.googleapis.com \
                       aiplatform.googleapis.com \
                       geocoding-backend.googleapis.com \
                       maps-backend.googleapis.com
```

### 2. Setup Local Environment

Clone the repository and set up the dependencies. If `uv` is not available in your Cloud Shell environment, you can use `pip` to install the dependencies in editable mode:

```bash
git clone https://github.com/google/adk-samples.git
cd adk-samples/python/agents/earth-engine-geospatial

# Using uv (recommended if available)
uv sync

# Fallback using pip
python3 -m venv .venv
./.venv/bin/pip install -e .
```

### 3. Generate API Key for Google Maps

Generate a new API key for the Google Maps Platform, restricting it to the required APIs for security:

```bash
gcloud services api-keys create --display-name="Earth Engine Agent Key" \
    --api-target=service=geocoding-backend.googleapis.com \
    --api-target=service=maps-backend.googleapis.com
```

To get the actual key string, list your keys and find the one you just created:

```bash
gcloud services api-keys list
```

Copy the key string and add it to your `.env` file as `GOOGLE_MAPS_API_KEY`.

### 4. Authentication

Authenticate your session:

```bash
gcloud auth application-default login
```

You may also need to run `earthengine authenticate` if prompted by the agent.

## Running the Agent

ADK provides convenient ways to bring up agents locally and interact with them.
You may talk to the agent using the CLI:

```bash
adk run earth_engine_geospatial
```

Or on a web interface:

```bash
adk web
```

In Google Cloud Shell, you need to handle CORS issues because the web interface is accessed via a Cloud Shell web preview URL. Run the agent using the following command to allow requests from Cloud Shell origins:

```bash
adk web --allow_origins 'regex:https://.*.cloudshell.dev'
```

### Convenience Script

If you need to restart the ADK web server, you can use the provided convenience script:

```bash
./restart_servers.sh
```

This script will kill any running instances of the ADK web server and start it again.

This flag tells the ADK web server to accept requests from your Cloud Shell instance, preventing CORS blocked errors in the browser.


Select `earth_engine_geospatial` from the dropdown.

## Example Interaction

Interact with the agent through a chat interface. The agent can answer basic
questions about land cover change in small to medium sized polygons represented
as GeoJSON strings. For example, here's a small polygon in the Santa Cruz
mountains of California, USA:

```
{"type":"Polygon","coordinates":[[[-122.25468153773132,37.21100075492321],
[-122.25468153773132,37.186046417670404],[-122.2224950295526,37.186046417670404],
[-122.2224950295526,37.21100075492321]]],"geodesic":false,"evenOdd":true}
```

Here is an [example script](https://code.earthengine.google.com/f81c949df0550ef68ea6aca3937ec9bd)
for generating GeoJSON for your area(s) of interest. Copy the GeoJSON printed
in the Console of the Earth Engine Code Editor to a chat:

User:

> Please tell me about
> {"type":"Polygon","coordinates":[[[-122.25468153773132,37.21100075492321],
> [-122.25468153773132,37.186046417670404],[-122.2224950295526,37.186046417670404],
> [-122.2224950295526, 37.21100075492321]]],"geodesic":false,"evenOdd":true}.

Agent:

> Based on an analysis of the provided geometry, here are the detected annual
> changes for the years 2020 and 2021:
>
> - 2020: 3.45 square kilometers of change were detected.
>
> - 2021: 2.74 square kilometers of change were detected.
>
> The provided coordinates are located in the Santa Cruz Mountains of California.
> The significant changes detected in 2020 are attributed to the CZU Lightning
> Complex fire, which burned a large area in this region in August 2020. The land
> cover transitioned from forest and shrubland to a burned landscape.
>
> The changes detected in 2021 represent post-fire effects, including vegetation
> regrowth and succession as the ecosystem begins to recover from the fire.

## Customization

This simple agent is extensible by providing more tools and more instructions
in the prompt. The tools can take this general form:

```python
@retry_async.AsyncRetry(deadline=60)
async def your_fancy_earth_engine_function(
    geojson: str,
) -> dict[str, Any]:
    """Gets some statistics about your area of interest (geojson).

    Args:
        geojson (str): A JSON string representing a GeoJSON geometry.

    Returns:
        A JSON dictionary.
    """
    region = ee.Geometry(json.loads(geojson))
    return await asyncio.to_thread(earth_engine_server_function(region).getInfo)
```

The `earth_engine_server_function` takes an `ee.Geometry` and returns an
`ee.Dictionary` (the output of a `reduceRegion()` call), both of which are
server variables. The `getInfo` call requests the result of the computation,
specifically the JSON representation of the `ee.Dictionary`. The function is
structured to make the request asynchronously and retry if it fails.

You can request textual representations of other server objects
(`ee.SomeObject`) using `getInfo()`. You can also request patches of pixels
as images. See [this guide](https://developers.google.com/earth-engine/guides/data_extraction)
for examples of programmatically extracting image data.

### Example: DOE National Lab Researcher

A researcher at a Department of Energy (DOE) national lab might want to analyze land surface temperature around energy infrastructure to study thermal impacts or site potential.

Here is an example of a new tool that computes the average land surface temperature using the MODIS dataset:

```python
@retry_async.AsyncRetry(deadline=60)
async def get_average_surface_temperature(
    geojson: str,
    start_date: str,
    end_date: str
) -> dict[str, Any]:
    """Gets the average land surface temperature within a geometry for a given date range.

    Args:
        geojson (str): A JSON string representing a GeoJSON geometry.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).

    Returns:
        A dictionary with the average temperature.
    """
    region = ee.Geometry(json.loads(geojson))
    collection = ee.ImageCollection("MODIS/061/MOD11A1").filterBounds(region).filterDate(start_date, end_date).select('LST_Day_1km')
    
    mean_image = collection.mean()
    # Scale factor for MODIS LST is 0.02
    mean_temp = mean_image.multiply(0.02).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=1000
    )
    return await asyncio.to_thread(mean_temp.getInfo)
```

To modify the agent, follow these steps to add the new tool and update the instructions:

#### 1. Add the Tool to `earth_engine_geospatial/tools.py`
Append the `get_average_surface_temperature` function to the end of the file.

#### 2. Register the Tool in `earth_engine_geospatial/agent.py`
Add `tools.get_average_surface_temperature` to the `tools` list in the `llm_agent.Agent` initialization:

```python
    tools=[
        tools.get_2017_2025_annual_changes,
        tools.generate_geojson_for_location,
        tools.generate_change_map_image,
        tools.create_interactive_map,
        tools.get_average_surface_temperature, # Add this line
    ],
```

#### 3. Update Prompt in `earth_engine_geospatial/prompt.py`
Replace the content of `prompt.py` with the following, which adds instructions for the new tool:

```python
root_agent_prompt = """
You are an expert geospatial analyst specializing in Google Earth Engine.
Use the `get_2017_2025_annual_changes` tool to detect annual changes in geometries.
Areas are provided to you as places, regions, or GeoJSON geometries.
If the user provides a location name instead of GeoJSON, use the `generate_geojson_for_location` tool to get the GeoJSON for that location.
The outputs from the `get_2017_2025_annual_changes` tool are a dictionary, keyed by year, with values of square meters of detected change in that year.
To visualize the change, first use the `generate_change_map_image` tool with the geometry to get a tile URL pattern. Then, use the `create_interactive_map` tool with that URL pattern to generate and save an interactive map HTML file as an artifact.
Use the coordinates in the geometry for additional factual evidence of land cover transitions reported to have occurred in the area for the change years.
To analyze surface temperature, use the `get_average_surface_temperature` tool. You must ask the user for a start date and end date if not provided.
Report the change years, change areas, temperature analysis, and the other evidence from your analysis to the user. Inform the user that the interactive map has been saved as an artifact.
"""
```

### Restart the Agent

Once you have made these changes, you need to restart the agent server to apply them. Use the provided convenience script:

```bash
./restart_servers.sh
```

This script will automatically shut down the running instance of the ADK web server and start it again with your new code.

## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended
for production use. It serves as a basic example of an agent and a foundational
starting point for individuals or teams to develop their own agents.

This sample has not been rigorously tested, may contain bugs or limitations,
and does not include features or optimizations typically required for a
production environment (e.g., robust error handling, security measures,
scalability, performance considerations, comprehensive logging, or advanced
configuration options).

Users are solely responsible for any further development, testing, security
hardening, and deployment of agents based on this sample. We recommend thorough
review, testing, and the implementation of appropriate safeguards before using
any derived agent in a live or critical system.
