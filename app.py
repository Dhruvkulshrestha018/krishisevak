import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Fetch the URL
mongo_url = os.getenv("MONGODB_URL")

# Debugging check: Ensure Python is actually reading the string
if not mongo_url:
    raise ValueError("❌ MONGODB_URL is missing from your .env file!")

# Connect to MongoDB
client = MongoClient(mongo_url)

try:
    response = client.admin.command("ping")
    print(" Pinged your deployment. You successfully connected to MongoDB!")
    print(f"Server Response: {response}")
except Exception as e:
    print(f"❌ Connection failed: {e}")