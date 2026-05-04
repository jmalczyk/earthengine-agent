# Earth Engine Geospatial Agent

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://console.cloud.google.com/cloudshell/editor?cloudshell_tutorial=README.md&ephemeral=false&cloudshell_git_repo=https://github.com/jmalczyk/earthengine-agent.git)

## Overview

This directory contains a [Google Earth Engine](https://earthengine.google.com/)
enabled ADK agent implemented as a simple chatbot.

The agent has tools to:
- Compute the area of annual change between 2017-2025 in a geometry provided by the user, using the [AlphaEarth Satellite Embeddings dataset](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL).
- Measure land cover area by type for years 2018-2025 using the [Dynamic World dataset](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) to provide more context about detected changes.
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

## Running the Agent

To install and run the agent copy and paste the following into the terminal to the bottom left of the page. 

```bash
./startup.sh
```

Once the script completes, click the Web Preview button in the top right corner of Cloud Shell to access the agent.

### Convenience Script

If you need to restart the ADK web server, you can use the provided convenience script:

```bash
./restart_servers.sh
```

This script will kill any running instances of the ADK web server and start it again.

This flag tells the ADK web server to accept requests from your Cloud Shell instance, preventing CORS blocked errors in the browser.


Select `earth_engine_geospatial` from the dropdown.

## Example Interaction

Interact with the agent through a chat interface. The agent can answer questions about land cover change in a region. You can specify a location name or provide a GeoJSON geometry.

For example, you can ask about a specific place:

User:

> Please tell me about the land cover changes in Santa Cruz, CA.

Agent:

> I have analyzed the area around Santa Cruz, CA.
>
> **Annual Change Detected (AlphaEarth):**
> - 2020: 5.2 sq km
> - 2021: 3.1 sq km
>
> **Land Cover Context (Dynamic World):**
> In 2020, there was a significant decrease in 'trees' and an increase in 'shrub_and_scrub' and 'bare' ground, suggesting a possible fire event (like the CZU Lightning Complex fire).
>
> I have generated an interactive map detailing the magnitude of change and saved it as an artifact `interactive_map.html`.

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

To modify the agent, follow these steps to add the new tool and update the instructions. We have added `"Paste changes here"` comment blocks in the relevant files to make this easier.

#### 1. Add the Tool to `earth_engine_geospatial/tools.py`
Append the `get_average_surface_temperature` function to the end of the file, where you see the `# --- PASTE NEW TOOLS HERE ---` comment.

#### 2. Register the Tool in `earth_engine_geospatial/agent.py`
Add `tools.get_average_surface_temperature` to the `tools` list in the `llm_agent.Agent` initialization, where you see the `# --- PASTE NEW TOOLS HERE ---` comment:

```python
    tools=[
        tools.get_2017_2025_annual_changes,
        tools.generate_geojson_for_location,
        tools.generate_change_map_image,
        tools.create_interactive_map,
        tools.get_dynamic_world_landcover_areas,
        # --- PASTE NEW TOOLS HERE ---
        tools.get_average_surface_temperature, # Add this line
    ],
```

#### 3. Update Prompt in `earth_engine_geospatial/prompt.py`
Replace the content of `prompt.py` with the following, or add instructions for the new tool where you see the `# --- PASTE NEW PROMPT INSTRUCTIONS HERE ---` comment:

```python
root_agent_prompt = """
You are an expert geospatial analyst specializing in Google Earth Engine.
Use the `get_2017_2025_annual_changes` tool to detect annual changes in geometries.
Areas are provided to you as places, regions, or GeoJSON geometries.
If the user provides a location name instead of GeoJSON, use the `generate_geojson_for_location` tool to get the GeoJSON for that location.
The outputs from the `get_2017_2025_annual_changes` tool are a dictionary, keyed by year, with values of square meters of detected change in that year.
Use the `get_dynamic_world_landcover_areas` tool to gather more context about the specific land cover types and their changes over time (2018-2025) within the geometry to explain the nature of the changes.
To visualize the change, first use the `generate_change_map_image` tool with the geometry to get a tile URL pattern. Then, use the `create_interactive_map` tool with that URL pattern to generate and save an interactive map HTML file as an artifact.
Use the coordinates in the geometry for additional factual evidence of land cover transitions reported to have occurred in the area for the change years.
To analyze surface temperature, use the `get_average_surface_temperature` tool. You must ask the user for a start date and end date if not provided.
Report the change years, change areas, temperature analysis, land cover transitions, and the other evidence from your analysis to the user. Inform the user that the interactive map has been saved as an artifact.
"""
```

### Restart the Agent

Once you have made these changes, you need to restart the agent server to apply them.

If the agent is running in the foreground of your terminal, you can stop it by pressing `Ctrl+C`.

Then, use the provided convenience script to restart it:

```bash
./restart_servers.sh
```

This script will automatically shut down any running background instances of the ADK web server and start it again with your new code.

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
