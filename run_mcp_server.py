import ee
import os
import sys
import dotenv

dotenv.load_dotenv()

# Initialize EE to prevent geeViz from printing interactive prompts to stdout
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "gmp-demos-483421")
try:
    ee.Initialize(project=project_id)
    print("Earth Engine initialized successfully in wrapper.", file=sys.stderr)
except Exception as e:
    print(f"Warning: Failed to initialize EE in wrapper: {e}", file=sys.stderr)

# Now run the MCP server
from geeViz.mcp.server import main

if __name__ == "__main__":
    main()
