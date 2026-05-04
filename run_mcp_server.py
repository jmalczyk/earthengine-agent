import ee
import os
import sys
import dotenv

dotenv.load_dotenv()

import google.auth
from google.oauth2 import service_account

# Initialize EE to prevent geeViz from printing interactive prompts to stdout
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "gmp-demos-483421")
cred_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

try:
    if cred_file:
        credentials = service_account.Credentials.from_service_account_file(cred_file)
        print(f"Using service account credentials from {cred_file}", file=sys.stderr)
    else:
        credentials, _ = google.auth.default()
        print("Using application default credentials", file=sys.stderr)
        
    ee.Initialize(credentials=credentials, project=project_id)
    print("Earth Engine initialized successfully in wrapper.", file=sys.stderr)
except Exception as e:
    print(f"Warning: Failed to initialize EE in wrapper: {e}", file=sys.stderr)

# Now run the MCP server
from geeViz.mcp.server import main

if __name__ == "__main__":
    main()
