import os
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# from dotenv import load_dotenv
from typing import Optional, Any

# load_dotenv(override=True)


class SimpleCosmosClient:
    def __init__(
        self,
        connection_string: str,
        database_name: str,
        partition_key_path: str,
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.partition_key_path = partition_key_path
        self.cosmos_client = None
        self.database_client = None
        self.container_client = None

    def connect(self) -> True:
        """
        Connects to the Cosmos DB account and gets the database client.
        """
        print("Connecting to Cosmos DB...")
        print(self.database_name)
        # print(self.connection_string)
        try:
            parts = self.connection_string.split(";")
            print(f"Connection string parts: {parts}")
            uri = None
            key = None
            for part in parts:
                if part.startswith("AccountEndpoint="):
                    uri = part.split("=")[1]
                elif part.startswith("AccountKey="):
                    key_start_index = part.find("=") + 1
                    key = part[key_start_index:]

            print(f"URI: {uri}")
            if not uri or not key:
                raise ValueError("Invalid connection string format")

            self.cosmos_client = CosmosClient(uri, key)
            print("CosmosClient initialized successfully...")

            self.database_client = self.cosmos_client.get_database_client(
                self.database_name
            )
            print(f"Database '{self.database_name}' client obtained.")

            return True

        except exceptions.CosmosResourceNotFoundError:
            print(
                f"Error: Database '{self.database_name}' not found. Please ensure the database name is correct and exists."
            )
            self.database_client = None
        except ValueError as e:
            print(f"Connection string error: {e}")
            self.cosmos_client = None
            self.database_client = None
        except Exception as e:
            print(f"An unexpected error occurred during connection: {e}")
            self.cosmos_client = None
            self.database_client = None
