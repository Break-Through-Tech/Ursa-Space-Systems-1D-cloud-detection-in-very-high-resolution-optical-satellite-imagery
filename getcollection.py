import geopandas as gpd
from pystac_client import Client

# Connect to the Maxar/Vantor Open Data STAC root
catalog_url = "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"
client = Client.open(catalog_url)

# Print all available disaster event collection IDs
# print("Available disaster events:")
# for collection in client.get_all_collections():
#    print(f"- {collection.id}")

# Target a specific event (e.g., Morocco Earthquake or a specific flood event)
# Replacing 'event-id' with an active collection name string listed above
collection_id = "HurricaneHelene-Oct24" 
collection = client.get_collection(collection_id)
