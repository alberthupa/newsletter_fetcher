import logging
import os
import json
import re
import base64
from datetime import datetime, timedelta, date as py_date
try:
    from datetime import timezone
except ImportError:
    timezone = None
import time
import tiktoken
from openai import OpenAI
from azure.cosmos import CosmosClient, exceptions

#from dotenv import load_dotenv
#load_dotenv()


try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    import html2text
    from email.utils import parsedate_to_datetime
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}")
    # Depending on deployment strategy, these might be installed via requirements.txt
    # or included in the function package. For now, proceed with placeholders.
    OpenAIEmbeddings = None
    Credentials = None
    Request = None
    build = None
    html2text = None
    parsedate_to_datetime = None


# --- Environment Variables & Constants ---
# These should be set in Azure Function App settings

DATABASE_NAME = "hupi-loch"  # Replace with your database name if different
CONTAINER_NAME = "knowledge-chunks"  # Replace with your container name




# --- Token handling functions (Copied from upload_old_newsletters.py) ---
def count_tokens(text: str) -> int:
    """Count the number of tokens in a text using the cl100k_base encoding."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


def chunk_text(text: str, max_tokens: int = 8000) -> list:
    """
    Split text into chunks that don't exceed the maximum token limit.
    (Copied from upload_old_newsletters.py)
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    current_chunk_tokens = []

    for token in tokens:
        current_chunk_tokens.append(token)

        if len(current_chunk_tokens) >= max_tokens:
            chunk_text = encoding.decode(current_chunk_tokens)
            chunks.append(chunk_text)
            current_chunk_tokens = []

    # Add any remaining tokens
    if current_chunk_tokens:
        chunk_text = encoding.decode(current_chunk_tokens)
        chunks.append(chunk_text)

    return chunks


# --- OpenAI Embeddings Client 
embeddings_client = None
if os.environ["OPENAI_API_KEY"]:
    try:
        class OpenAIEmbeddings:
            """
            sth = OpenAIEmbeddings().get_openai_embedding("test")
            print(sth.data[0].embedding)
            """

            #def set_embeddings_client(self):
            #    return OpenAI()

            def get_openai_embedding(self, text):
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                return client.embeddings.create(input=text, model="text-embedding-3-small")
            
        embeddings_client = OpenAIEmbeddings()

        logging.info("OpenAIEmbeddings client initialized.")
    except Exception as e:
        logging.error(f"Error initializing OpenAIEmbeddings client: {e}")
else:
    logging.warning(
        "OpenAIEmbeddings client not initialized. OPENAI_API_KEY or OpenAIEmbeddings class missing."
    )


# --- Cosmos DB Client (Copied and adapted from upload_old_newsletters.py) ---
def get_cosmos_client(connection_string):
    """Initializes and returns a CosmosClient."""
    if not connection_string:
        logging.error("Error: COSMOS_CONNECTION_STRING is not set.")
        return None
    try:
        # Assuming connection string format "AccountEndpoint=...;AccountKey=..."
        parts = connection_string.split(";")
        uri = None
        key = None
        for part in parts:
            if part.startswith("AccountEndpoint="):
                uri = part.split("=")[1]
            elif part.startswith("AccountKey="):
                key_start_index = part.find("=") + 1
                key = part[key_start_index:]

        if not uri or not key:
            raise ValueError(
                "Invalid connection string format. Ensure AccountEndpoint and AccountKey are present."
            )
        logging.info(f"Attempting to connect to Cosmos DB at URI: {uri}")
        client = CosmosClient(uri, credential=key)
        logging.info("CosmosClient initialized successfully.")
        return client
    except ValueError as ve:
        logging.error(f"Error initializing CosmosClient: {ve}")
        return None
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while initializing CosmosClient: {e}"
        )
        return None


def get_container_client(client, database_name, container_name):
    """Gets a reference to the database and container."""
    if not client:
        logging.error("Error: Cosmos client is not initialized.")
        return None
    try:
        database = client.get_database_client(database_name)
        logging.info(f"Successfully got database client for '{database_name}'.")
        container = database.get_container_client(container_name)
        logging.info(f"Successfully got container client for '{container_name}'.")
        logging.info(
            f"Successfully connected to database '{database_name}' and container '{container_name}'."
        )
        return container
    except exceptions.CosmosResourceNotFoundError as e:
        if f"dbs/{database_name}" in str(e):
            logging.error(f"Error: Database '{database_name}' not found.")
        elif f"colls/{container_name}" in str(e):
            logging.error(
                f"Error: Container '{container_name}' not found in database '{database_name}'."
            )
        else:
            logging.error(f"Error: A Cosmos DB resource was not found: {e}")
        return None
    except exceptions.CosmosHttpResponseError as e:
        logging.error(f"An unexpected Cosmos DB HTTP error occurred: {e}")
        return None
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while getting container client: {e}"
        )
        return None


def check_item_exists(container, item_id):
    """Checks if an item with the given ID exists in the container."""
    if not container:
        logging.debug(f"Debug: Container not available for checking item {item_id}.")
        return False
    try:
        logging.debug(
            f"Debug: Checking if item with id '{item_id}' exists in Cosmos DB..."
        )
        # Assuming partition key is the same as item ID
        container.read_item(item=item_id, partition_key=item_id)
        logging.debug(f"Debug: Item with id '{item_id}' FOUND in Cosmos DB.")
        return True
    except exceptions.CosmosResourceNotFoundError:
        logging.debug(f"Debug: Item with id '{item_id}' NOT FOUND in Cosmos DB.")
        return False
    except exceptions.CosmosHttpResponseError as e:
        logging.debug(
            f"Debug: Error checking item '{item_id}': {e}. Assuming not found."
        )
        return False
    except Exception as e:
        logging.debug(
            f"Debug: Unexpected error checking item '{item_id}': {e}. Assuming not found."
        )
        return False


def upload_text_chunk(
    container,
    item_id,
    text_content,
    tags,
    vector_embedding=None,
    custom_properties=None,
):
    """Uploads a text chunk with tags and custom properties to the container."""
    if not container:
        logging.debug(f"Debug: Container not available for uploading item {item_id}.")
        return
    document = {
        "id": item_id,
        "text": text_content,
        "tags": tags,
        "embedding": vector_embedding,
    }
    if custom_properties:
        document.update(custom_properties)

    logging.debug(f"Debug: Preparing to upload document with id: {item_id}")

    try:
        container.upsert_item(document)
        logging.debug(f"Debug: Successfully uploaded/updated item with id: {item_id}")
    except exceptions.CosmosHttpResponseError as e:
        logging.error(f"Debug: Error uploading item {item_id}: {e}")
        logging.error(
            f"Debug: Failed document details: {json.dumps(document, default=str)}"
        )
    except Exception as e:
        logging.error(f"Debug: Unexpected error uploading item {item_id}: {e}")


# --- Google API Client (Copied and adapted from upload_old_newsletters.py) ---
def get_gmail_service():
    """Initializes and returns the Gmail API service."""
    logging.info("Debug: Initializing Gmail service...")
    if not all([os.environ["GMAIL_REFRESH_TOKEN"], os.environ["GMAIL_CLIENT_ID"], os.environ["GMAIL_CLIENT_SECRET"]]):
        logging.error(
            "Error: Gmail API credentials (refresh token, client ID, client secret) not fully set."
        )
        return None
    if not Credentials or not Request or not build:
        logging.error("Error: Required Google API client libraries not imported.")
        return None
    try:
        gmail_creds = Credentials(
            token=None,
            refresh_token=os.environ["GMAIL_REFRESH_TOKEN"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.environ["GMAIL_CLIENT_ID"],
            client_secret=os.environ["GMAIL_CLIENT_SECRET"],
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        gmail_creds.refresh(Request())
        service = build("gmail", "v1", credentials=gmail_creds)
        logging.info("Debug: Gmail service initialized successfully.")
        return service
    except Exception as e:
        logging.error(f"Error initializing Gmail service: {e}")
        return None


# --- Email Processing Functions (Copied from upload_old_newsletters.py) ---
def clean_markdown(md):
    """Cleans markdown text extracted from HTML emails."""
    # (Paste the entire clean_markdown function content here)
    md = re.sub(r"!\[\]\(https://track\.[^\)]+\)", "", md)
    md = re.sub(r"\[View in browser\]\([^\)]+\)", "", md, flags=re.I)
    md = re.sub(
        r"\[.*?\]\((https?://(?:track\.aisecret\.us|click\.convertkit-mail2\.com|mandrillapp\.com|clicks\.mlsend\.com|click\.sender\.net|t\.dripemail2\.com|click\.revue\.email|ct\.beehiiv\.com|clicks\.aweber\.com|hubspotlinks\.com|getresponse\.com|substack\.com|mailerlite\.com|sendgrid\.net|sparkpostmail\.com|amazonseS\.com)[^\)]+)\)",
        "",
        md,
    )
    md = re.sub(r"\[!\[\]\([^\)]+\)\]\([^\)]+\)", "", md)
    md = re.sub(r"^\s*> \[.*?SPONSORED.*?\]\(.*?\)\s*$", "", md, flags=re.M | re.I)

    md = re.sub(r"\[(.*?)\]\([^\)]+\)", r"\1", md)

    md = re.sub(r"\*\s?\*\s?\*", "", md)
    md = re.sub(r"---+", "", md)
    md = re.sub(r"\|\s?.*?\s?\|", "", md)

    unsubscribe_patterns = [
        r"https://track\.aisecret\.us/track/unsubscribe\.do\?",
        r"If you wish to stop receiving our emails.*?click here",
        r"To unsubscribe from this newsletter.*?click here",
        r"No longer want to receive these emails\?",
        r"Unsubscribe here",
        r"Manage your preferences",
        r"Update your profile",
    ]
    for pattern in unsubscribe_patterns:
        md = re.split(pattern, md, flags=re.I)[0].strip()

    # Fix isolated characters between newlines (like "\n\n!\n\n")
    md = re.sub(
        r"\n\n([^\w\s]{1,3})\n\n", r" \1 ", md
    )  # Handle isolated punctuation/symbols
    md = re.sub(r"\n\n([a-zA-Z])\n\n", r" \1 ", md)  # Handle isolated single letters
    md = re.sub(
        r"(\w)\n\n([^\w\s]{1,2})\n\n(\w)", r"\1 \2 \3", md
    )  # Words with punctuation between

    # General newline cleanup
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r" +\n", "\n", md)
    md = re.sub(r"\n +", "\n", md)

    # Fix remaining adjacent newlines with single character in between
    md = re.sub(r"(\w)\n([^\w\s])\n(\w)", r"\1 \2 \3", md)

    # Remove isolated newlines that should be spaces
    md = re.sub(r"([a-zA-Z])\n([a-zA-Z])", r"\1 \2", md)

    md = re.sub(r"^\s*\[image:.*?\]\s*$", "", md, flags=re.M | re.I)
    md = re.sub(r"^\s*!\[.*?\]\(.*?\)\s*$", "", md, flags=re.M | re.I)
    md = re.sub(r"^\s*[-=_\*#]{3,}\s*$", "", md, flags=re.M)

    return md.strip()


def find_html_part(parts):
    """Recursively finds the HTML part in email message parts."""
    # (Paste the entire find_html_part function content here)
    for part in parts:
        if part.get("mimeType") == "text/html" and part.get("body", {}).get("data"):
            return part["body"]["data"]
        if "parts" in part:
            html_data = find_html_part(part["parts"])
            if html_data:
                return html_data
    return None


def process_newsletters_for_date(
    target_date_str, gmail_service, container_client, embeddings_client_instance
):
    """
    Processes newsletters for a specific date.
    (Copied and adapted from upload_old_newsletters.py)
    """
    logging.info(f"Debug: Starting newsletter processing for date: {target_date_str}")

    try:
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        logging.error(
            f"Error: Invalid date format '{target_date_str}'. Please use YYYY-MM-DD."
        )
        return

    date_after = target_dt.strftime("%Y/%m/%d")
    date_before = (target_dt + timedelta(days=1)).strftime("%Y/%m/%d")
    query = f"after:{date_after} before:{date_before}"
    logging.info(f"Debug: Gmail search query: '{query}'")

    try:
        results = (
            gmail_service.users()
            .messages()
            .list(
                userId="me", q=query, labelIds=["Label_58"], maxResults=50
            )  # Assuming Label_58 is your newsletter label
            .execute()
        )
    except Exception as e:
        logging.error(f"Error querying Gmail: {e}")
        return

    messages = results.get("messages", [])
    logging.info(f"Debug: Found {len(messages)} messages for {target_date_str}.")

    if not messages:
        logging.info(f"Debug: No messages found for {target_date_str}. Exiting.")
        return

    for msg_summary in messages:
        msg_id = msg_summary["id"]
        logging.info(f"\nDebug: Processing message ID: {msg_id}")

        if check_item_exists(container_client, msg_id):
            logging.info(
                f"Debug: Message ID {msg_id} already exists in Cosmos DB. Skipping."
            )
            continue

        try:
            full_msg = (
                gmail_service.users()
                .messages()
                .get(userId="me", id=msg_id, format="full")
                .execute()
            )
        except Exception as e:
            logging.error(f"Error fetching full message for ID {msg_id}: {e}")
            continue

        payload = full_msg.get("payload", {})
        headers = {h["name"]: h["value"] for h in payload.get("headers", [])}

        subject = headers.get("Subject", "No Subject")
        sender = headers.get("From", "Unknown Sender")
        raw_date_header = headers.get("Date", "")

        try:
            # Ensure parsedate_to_datetime is available
            if parsedate_to_datetime:
                parsed_date_obj = parsedate_to_datetime(raw_date_header)
                message_date_iso = parsed_date_obj.date().isoformat()
            else:
                logging.warning(
                    "parsedate_to_datetime not available. Using target_date_str."
                )
                message_date_iso = target_date_str
        except Exception:
            message_date_iso = target_date_str

        logging.info(f"Debug: Subject: {subject}")
        logging.info(f"Debug: From: {sender}")
        logging.info(f"Debug: Date: {message_date_iso}")

        html_data = None
        if payload.get("mimeType") == "text/html":
            html_data = payload.get("body", {}).get("data")
        elif "parts" in payload:
            html_data = find_html_part(payload["parts"])

        if not html_data:
            logging.info(
                f"Debug: No HTML content found for message ID {msg_id}. Skipping."
            )
            continue

        try:
            decoded_html = base64.urlsafe_b64decode(html_data.encode("UTF-8")).decode(
                "UTF-8"
            )
        except Exception as e:
            logging.error(
                f"Debug: Error decoding HTML for message ID {msg_id}: {e}. Skipping."
            )
            continue

        # Ensure html2text is available
        if html2text:
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.body_width = 0
            markdown_body = h.handle(decoded_html)
            cleaned_body = clean_markdown(markdown_body)
        else:
            logging.warning(
                "html2text not available. Skipping HTML to Markdown conversion."
            )
            cleaned_body = decoded_html  # Use raw HTML if conversion fails

        if not cleaned_body.strip():
            logging.info(
                f"Debug: Cleaned body is empty for message ID {msg_id}. Skipping."
            )
            continue

        logging.info(
            f"Debug: Cleaned content (first 100 chars): {cleaned_body[:100].replace(chr(10), ' ')}..."
        )

        text_to_embed = f"Subject: {subject}\n\n{cleaned_body}"
        token_count = count_tokens(text_to_embed)
        logging.info(f"Debug: Text has {token_count} tokens")

        text_chunks_list = (
            chunk_text(text_to_embed) if token_count > 8000 else [text_to_embed]
        )
        logging.info(f"Debug: Text split into {len(text_chunks_list)} chunks")

        for chunk_index, chunk_content in enumerate(text_chunks_list):
            chunk_msg_id = (
                msg_id
                if len(text_chunks_list) == 1
                else f"{msg_id}_chunk_{chunk_index}"
            )

            if check_item_exists(container_client, chunk_msg_id):
                logging.info(
                    f"Debug: Chunk ID {chunk_msg_id} already exists in Cosmos DB. Skipping."
                )
                continue

            calculated_embeddings = None
            if embeddings_client_instance:
                try:
                    logging.info(
                        f"Debug: Calculating embeddings for chunk {chunk_index+1}/{len(text_chunks_list)}..."
                    )
                    embeddings_result = embeddings_client_instance.get_openai_embedding(
                        chunk_content
                    )
                    calculated_embeddings = embeddings_result.data[0].embedding
                    logging.info(
                        f"Debug: Embeddings calculated successfully for chunk {chunk_index+1}."
                    )
                except Exception as e:
                    logging.error(
                        f"Error calculating embeddings for chunk {chunk_index+1}: {e}"
                    )
                    calculated_embeddings = None
            else:
                logging.warning(
                    "Debug: Embeddings client not available. Skipping embedding calculation."
                )

            custom_props = {
                "source": "gmail_newsletter",
                "subject": subject,
                "author": sender,
                "chunk_date": message_date_iso,
                # "processing_target_date": target_date_str, # Removed as it might be confusing in a daily run
                "gmail_message_id": msg_id,
                "ingestion_date": datetime.utcnow().isoformat() + "Z",
            }

            if len(text_chunks_list) > 1:
                custom_props.update(
                    {
                        "is_chunk": True,
                        "chunk_index": chunk_index,
                        "total_chunks": len(text_chunks_list),
                        "original_id": msg_id,
                    }
                )

            upload_text_chunk(
                container_client,
                item_id=chunk_msg_id,
                text_content=chunk_content.replace("Subject: ", "", 1)
                if chunk_content.startswith("Subject: ")
                else chunk_content,
                tags=[
                    "newsletter",
                    sender,
                    subject[:50],
                ],  # Consider sanitizing sender/subject for tags
                vector_embedding=calculated_embeddings,
                custom_properties=custom_props,
            )
            logging.info(
                f"Debug: Upload process initiated for chunk {chunk_index+1}/{len(text_chunks_list)}."
            )


# --- Function to get the last newsletter date (Based on user's snippet) ---
def get_last_newsletter_date(container_client):
    """
    Queries Cosmos DB to find the maximum chunk_date for gmail_newsletter source.
    Returns a datetime.date object or None if no date is found.
    """
    if not container_client:
        logging.error(
            "Cosmos container client not available to get last newsletter date."
        )
        return None

    query = f"SELECT VALUE max(c.chunk_date) FROM c WHERE c.source = 'gmail_newsletter'"
    results = []
    try:
        # Assuming query_items returns an iterable
        for item in container_client.query_items(
            query=query, enable_cross_partition_query=True
        ):
            results.append(item)
        logging.info(f"Found {len(results)} results for last date query.")

        if results and results[0] is not None:
            # The result is expected to be a string in 'YYYY-MM-DD' format
            last_date_str = results[0]
            try:
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
                logging.info(f"Last newsletter date found: {last_date}")
                return last_date
            except ValueError:
                logging.error(
                    f"Failed to parse last date string from Cosmos DB: {last_date_str}"
                )
                return None
        else:
            logging.info("No last newsletter date found in Cosmos DB.")
            return None

    except exceptions.CosmosHttpResponseError as e:
        logging.error(f"Error during last date query: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during last date query: {e}")
        return None



if __name__ == "__main__":
    print("--- Starting Newsletter Ingestion Script (Historical Processing) ---")
    if 'timezone' in globals() and timezone is not None:
        utc_timestamp = datetime.now(timezone.utc).isoformat()
    else:
        utc_timestamp = datetime.utcnow().isoformat() + "Z"



    logging.info("Python timer trigger function started at %s", utc_timestamp)

    # Initialize clients
    cosmos_client_instance = get_cosmos_client(os.environ["COSMOS_CONNECTION_STRING"])
    container_client_instance = None
    if cosmos_client_instance:
        container_client_instance = get_container_client(
            cosmos_client_instance, DATABASE_NAME, CONTAINER_NAME
        )

    gmail_service_instance = get_gmail_service()

    # Check if all necessary clients are initialized
    if not all([gmail_service_instance, container_client_instance, embeddings_client]):
        logging.error(
            "One or more services (Gmail, Cosmos DB, OpenAI Embeddings) could not be initialized. Exiting function."
        )
        if not gmail_service_instance:
            logging.error("Gmail service failed to initialize.")
        if not container_client_instance:
            logging.error("Cosmos container client failed to initialize.")
        if not embeddings_client:
            logging.warning(
                "OpenAI embeddings client not initialized. Embeddings will be skipped."
            )
    else:
        # Determine the date range for ingestion
        last_downloaded_date = get_last_newsletter_date(container_client_instance)
        today = py_date.today()

        
        if last_downloaded_date:
            # Start from the day after the last downloaded date
            start_date = last_downloaded_date + timedelta(days=1)
            logging.info(
                f"Last downloaded date found: {last_downloaded_date}. Starting ingestion from {start_date}."
            )
        else:
            # Start from today minus 7 days as per user's preference
            start_date = today - timedelta(days=7)
            logging.info(
                f"No last downloaded date found. Starting initial ingestion from {start_date} (today - 7 days)."
            )

        end_date = today  # Ingest up to and including today

        logging.info(f"Ingestion date range: {start_date} to {end_date}")

        '''

        # Iterate through the date range and process newsletters for each day
        current_date = start_date
        while current_date <= end_date:
            target_date_str = current_date.strftime("%Y-%m-%d")
            logging.info(f"\n--- Processing date: {target_date_str} ---")

            process_newsletters_for_date(
                target_date_str,
                gmail_service_instance,
                container_client_instance,
                embeddings_client,  # Pass the embeddings client instance
            )

            current_date += timedelta(days=1)
            # No need for sleep in Azure Function timer trigger between dates

        logging.info(
            "Python timer trigger function finished at %s",
            datetime.utcnow().replace(tzinfo=None).isoformat(),
        )
        ''' 