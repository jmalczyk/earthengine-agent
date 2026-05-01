"""Earth Engine agent tools."""

import asyncio
import base64
from google.genai import types
import json
import os
from typing import Any

import requests


import ee
import numpy as np
from google.api_core import retry_async


def get_angle(image1: ee.Image, image2: ee.Image) -> ee.Image:
    """Calculates the angle between two Earth Engine images.

    This function treats each pixel as a vector and computes the angle between
    the vectors from `image1` and `image2` using the dot product formula:
    angle = acos((image1 * image2) / (|image1| * |image2|)).
    Assuming the images are already normalized or the magnitude is handled
    elsewhere, this implementation simplifies to acos(dot_product).

    Args:
        image1: The first ee.Image.
        image2: The second ee.Image.

    Returns:
        An ee.Image containing the angle in radians.
    """
    return (
        image1.multiply(image2).reduce(ee.Reducer.sum()).acos().rename("angle")
    )


def get_change_year_image(threshold: float):
    """Generates an image showing the year of significant change.

    This function uses the GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL dataset to detect
    significant changes between consecutive years from 2018 to 2025. It calculates
    the angle between the embeddings of each year and the previous year. Pixels
    where this angle exceeds pi/4 are considered to have undergone a significant
    change. The output is a multi-band image where each band corresponds to a
    year, and the pixel value is the year if a significant change was detected
    in that year compared to the previous one.

    Args:
        threshold: Angular threshold in radians above which change is assumed.

    Returns:
        An ee.Image with bands for each year from 2018 to 2025, indicating
        the year of change.
    """
    embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    years = ee.List.sequence(2018, 2025)

    def annual_changes(year: int) -> ee.Image:
        cur = embeddings.filter(
            ee.Filter.calendarRange(year, year, "year")
        ).mosaic()
        prev_year = ee.Number(year).subtract(1)
        prev = embeddings.filter(
            ee.Filter.calendarRange(prev_year, prev_year, "year")
        ).mosaic()
        return (
            get_angle(prev, cur)
            .gt(threshold)
            .multiply(ee.Image.constant(year))
            .selfMask()
            .rename(ee.Number(year).format("%d"))
        )

    changes = (
        ee.ImageCollection.fromImages(years.map(annual_changes))
        .toBands()
        .rename(years.map(lambda s: ee.Number(s).format("%d")))
    )
    return changes


def get_annual_change_dictionary(geometry: ee.Geometry, scale: int = 10) -> ee.Dictionary:
    """Gets a dictionary of annual change areas within a given geometry.

    This function calculates the total area (in square meters) for each year
    (from 2018 to 2024) where significant land cover change was detected within
    the specified Earth Engine geometry.

    Args:
        geometry: The ee.Geometry in which to compute the change areas.
        scale: The scale in meters to use for the reduction. Defaults to 10.

    Returns:
        An ee.Dictionary where keys are years (as strings) and values are the
        total area in square meters for which change was detected in that year.
    """
    threshold = np.pi / 4  # Arbitrary.
    change_year_image = get_change_year_image(threshold)
    change_year_areas = change_year_image.gt(0).multiply(ee.Image.pixelArea())
    return change_year_areas.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geometry, scale=scale, maxPixels=1e13
    )


def get_change_magnitude_image() -> ee.Image:
    """Generates an image showing the maximum magnitude of change.

    This function uses the GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL dataset to detect
    changes between consecutive years from 2018 to 2025. It calculates the angle
    between the embeddings of each year and the previous year. The output is an
    image where each pixel value is the maximum angle (magnitude of change)
    detected across all year pairs.

    Returns:
        An ee.Image with a single band 'magnitude' containing the maximum angle.
    """
    embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    years = ee.List.sequence(2018, 2025)

    def annual_angles(year: int) -> ee.Image:
        cur = embeddings.filter(
            ee.Filter.calendarRange(year, year, "year")
        ).mosaic()
        prev_year = ee.Number(year).subtract(1)
        prev = embeddings.filter(
            ee.Filter.calendarRange(prev_year, prev_year, "year")
        ).mosaic()
        return get_angle(prev, cur).rename(ee.Number(year).format("%d"))

    angles = (
        ee.ImageCollection.fromImages(years.map(annual_angles))
        .toBands()
    )
    return angles.reduce(ee.Reducer.max()).rename("magnitude")


@retry_async.AsyncRetry(deadline=60)
async def generate_change_map_image(
    geometry: ee.Geometry | str,
) -> str:
    """Generates an Earth Engine XYZ tile URL pattern detailing the magnitude of change within the area.

    Args:
        geometry (ee.Geometry | str): An Earth Engine geometry object or a JSON string representing a GeoJSON geometry.

    Returns:
        str: A URL pattern pointing to XYZ tiles of the change magnitude.
    """
    if isinstance(geometry, str):
        region = ee.Geometry(json.loads(geometry))
    else:
        region = geometry

    def _call():
        magnitude_image = get_change_magnitude_image()
        
        viz_params = {
            'min': 0,
            'max': 1.5,
            'palette': ['blue', 'cyan', 'green', 'yellow', 'red'],
        }
        
        map_id_dict = magnitude_image.getMapId(viz_params)
        return map_id_dict['tile_fetcher'].url_format

    return await asyncio.to_thread(_call)


async def create_interactive_map(tile_url_pattern: str, tool_context: Any, center_lat: float = 0, center_lng: float = 0, zoom: int = 2) -> str:
    """Generates an HTML page with a Google Map and controls, and saves it as an artifact.

    Args:
        tile_url_pattern (str): The XYZ tile URL pattern.
        tool_context (Any): The tool context injected by ADK.
        center_lat (float): Latitude to center the map.
        center_lng (float): Longitude to center the map.
        zoom (int): Initial zoom level.

    Returns:
        str: A message indicating the map was saved as an artifact.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set.")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Interactive Change Map</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        html, body {{
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: 'Google Sans', Roboto, sans-serif;
            background-color: #151316;
            color: #e6e1e6;
        }}
        #map {{
            height: 100%;
            width: 100%;
        }}
        .control-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(33, 31, 34, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            width: 220px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .control-panel h3 {{
            margin-top: 0;
            font-size: 18px;
            color: #d5baff;
            margin-bottom: 15px;
        }}
        .control-group {{
            margin-bottom: 15px;
        }}
        .control-group label {{
            display: block;
            font-size: 12px;
            color: #cec2db;
            margin-bottom: 5px;
        }}
        .control-group input[type="range"] {{
            width: 100%;
            accent-color: #7cc4ff;
        }}
        .control-group input[type="checkbox"] {{
            margin-right: 8px;
            accent-color: #7cc4ff;
        }}
    </style>
    <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap" async defer></script>
    <script>
        var map;
        var tileLayer;

        function initMap() {{
            map = new google.maps.Map(document.getElementById('map'), {{
                center: {{lat: {center_lat}, lng: {center_lng}}},
                zoom: {zoom},
                mapTypeId: 'roadmap',
                styles: [
                    {{
                        "elementType": "geometry",
                        "stylers": [{{ "color": "#212121" }}]
                    }},
                    {{
                        "elementType": "labels.icon",
                        "stylers": [{{ "visibility": "off" }}]
                    }},
                    {{
                        "elementType": "labels.text.fill",
                        "stylers": [{{ "color": "#757575" }}]
                    }},
                    {{
                        "elementType": "labels.text.stroke",
                        "stylers": [{{ "color": "#212121" }}]
                    }},
                    {{
                        "featureType": "administrative",
                        "elementType": "geometry",
                        "stylers": [{{ "color": "#757575" }}]
                    }},
                    {{
                        "featureType": "administrative.country",
                        "elementType": "labels.text.fill",
                        "stylers": [{{ "color": "#9e9e9e" }}]
                    }},
                    {{
                        "featureType": "water",
                        "elementType": "geometry",
                        "stylers": [{{ "color": "#000000" }}]
                    }},
                    {{
                        "featureType": "water",
                        "elementType": "labels.text.fill",
                        "stylers": [{{ "color": "#3d3d3d" }}]
                    }}
                ]
            }});

            var tileUrlPattern = "{tile_url_pattern}";

            tileLayer = new google.maps.ImageMapType({{
                getTileUrl: function(coord, zoom) {{
                    return tileUrlPattern
                        .replace('{{z}}', zoom)
                        .replace('{{x}}', coord.x)
                        .replace('{{y}}', coord.y);
                }},
                tileSize: new google.maps.Size(256, 256),
                name: 'Change Overlay',
                maxZoom: 20,
                minZoom: 0
            }});

            map.overlayMapTypes.push(tileLayer);

            document.getElementById('opacity-slider').addEventListener('input', function(e) {{
                tileLayer.setOpacity(parseFloat(e.target.value));
            }});

            document.getElementById('visibility-toggle').addEventListener('change', function(e) {{
                if (e.target.checked) {{
                    map.overlayMapTypes.push(tileLayer);
                }} else {{
                    var index = map.overlayMapTypes.indexOf(tileLayer);
                    if (index > -1) {{
                        map.overlayMapTypes.removeAt(index);
                    }}
                }}
            }});
        }}
    </script>
</head>
<body>
    <div id="map"></div>
    <div class="control-panel">
        <h3>Layer Controls</h3>
        <div class="control-group">
            <label for="opacity-slider">Opacity</label>
            <input type="range" id="opacity-slider" min="0" max="1" step="0.1" value="1">
        </div>
        <div class="control-group">
            <input type="checkbox" id="visibility-toggle" checked>
            <label for="visibility-toggle" style="display:inline;">Show Overlay</label>
        </div>
    </div>
</body>
</html>
"""
    
    html_bytes = html_content.encode('utf-8')
    await tool_context.save_artifact(
        filename="interactive_map.html",
        artifact=types.Part(
            inline_data=types.Blob(mime_type="text/html", data=html_bytes)
        ),
    )
    return "Interactive map saved as artifact 'interactive_map.html'"


@retry_async.AsyncRetry(deadline=60)
async def get_2017_2025_annual_changes(
    geometry: ee.Geometry | str,
) -> dict[str, Any]:
    """Gets a dictionary of annual change areas within a given geometry.

    This function calculates the total area (in square meters) for each year
    (from 2018 to 2025) where significant land cover change was detected within
    the specified geometry.

    Args:
        geometry (ee.Geometry | str): An Earth Engine geometry object or a JSON string representing a GeoJSON geometry.

    Returns:
        A dictionary where keys are years (as strings) and values are the
        total area in square meters for which change was detected in that year.
    """
    if isinstance(geometry, str):
        region = ee.Geometry(json.loads(geometry))
    else:
        region = geometry

    scale = 10
    while scale <= 1000:
        try:
            return await asyncio.to_thread(get_annual_change_dictionary(region, scale).getInfo)
        except Exception as e:
            if "User memory limit exceeded" in str(e):
                print(f"Memory limit exceeded at scale {scale}. Retrying with scale {scale * 2}...")
                scale *= 2
            else:
                raise e
    raise RuntimeError("Exceeded maximum scale for computation.")


async def generate_geojson_for_location(location: str) -> str:
    """Generates a GeoJSON Polygon for a given location name using Google Maps Geocoding API.

    Args:
        location (str): The name of the location (e.g., "Santa Cruz, CA").

    Returns:
        str: A JSON string representing a GeoJSON Polygon.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set.")

    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={api_key}"

    def _call():
        response = requests.get(url)
        data = response.json()
        if data["status"] == "OK":
            result = data["results"][0]
            bounds = result["geometry"].get("bounds") or result["geometry"]["viewport"]
            northeast = bounds["northeast"]
            southwest = bounds["southwest"]
            geojson = {
                "type": "Polygon",
                "coordinates": [[
                    [southwest["lng"], southwest["lat"]],
                    [northeast["lng"], southwest["lat"]],
                    [northeast["lng"], northeast["lat"]],
                    [southwest["lng"], northeast["lat"]],
                    [southwest["lng"], southwest["lat"]]
                ]]
            }
            return json.dumps(geojson)
        else:
            raise Exception(f"Failed to geocode location: {data['status']}")

    return await asyncio.to_thread(_call)


async def generate_geometry_for_location(location: str) -> ee.Geometry:
    """Generates an Earth Engine Geometry for a given location name.

    Args:
        location (str): The name of the location (e.g., "Santa Cruz, CA").

    Returns:
        ee.Geometry: An Earth Engine Geometry object.
    """
    geojson_str = await generate_geojson_for_location(location)
    return ee.Geometry(json.loads(geojson_str))





