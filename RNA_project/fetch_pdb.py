import requests
import json
import os
from datetime import datetime

def get_latest_rna_structures(limit=50):
    """
    Fetches a pool of the latest PDB entries containing RNA.
    
    Returns:
        list: A list of dictionaries containing PDB ID, Title, and Release Date.
    """
    
    # 1. SEARCH API ENDPOINT
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

    # Define the search query
    # We look for entries where the RNA polymer count is >= 1.
    # We sort by 'rcsb_accession_info.initial_release_date' in descending order.
    query_payload = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                "operator": "greater_or_equal",
                "value": 1
            }
        },
        "request_options": {
            "sort": [
                {
                    "sort_by": "rcsb_accession_info.initial_release_date",
                    "direction": "desc"
                }
            ],
            "paginate": {
                "start": 0,
                "rows": limit
            }
        },
        "return_type": "entry"
    }

    try:
        response = requests.post(search_url, json=query_payload)
        response.raise_for_status()
        results = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Search API: {e}")
        return []

    # Extract PDB IDs from the search result
    pdb_ids = [result['identifier'] for result in results.get('result_set', [])]
    
    if not pdb_ids:
        print("No RNA structures found.")
        return []

    # 2. DATA API (GraphQL)
    # We use GraphQL to fetch metadata for all IDs in a single request.
    data_url = "https://data.rcsb.org/graphql"

    graphql_query = """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        struct {
          title
        }
        rcsb_accession_info {
          initial_release_date
        }
      }
    }
    """

    variables = {"ids": pdb_ids}

    try:
        data_response = requests.post(data_url, json={'query': graphql_query, 'variables': variables})
        data_response.raise_for_status()
        data_results = data_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Data API: {e}")
        return []

    # Process and format the output
    entries = data_results.get('data', {}).get('entries', [])
    formatted_results = []

    for entry in entries:
        # Safely get fields
        pdb_id = entry.get('rcsb_id')
        title = entry.get('struct', {}).get('title', 'No Title')
        date_str = entry.get('rcsb_accession_info', {}).get('initial_release_date', '')
        
        # Format date for readability if present
        if date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            date_display = dt.strftime('%Y-%m-%d')
        else:
            date_display = "N/A"

        formatted_results.append({
            "id": pdb_id,
            "date": date_display,
            "title": title
        })

    # Sort again by date (just in case GraphQL returned them out of order)
    formatted_results.sort(key=lambda x: x['date'], reverse=True)
    
    return formatted_results

def download_only_pdb_structures(structures, save_dir, target_count=10):
    """
    Downloads .pdb files.
    Skips entries that do not have a .pdb file (e.g. large structures only available in .cif).
    Stops when target_count is reached.
    
    Args:
        structures (list): List of dicts with 'id' keys.
        save_dir (str): Directory path to save files.
        target_count (int): How many successful downloads we want.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        except OSError as e:
            print(f"Error creating directory {save_dir}: {e}")
            return

    base_url = "https://files.rcsb.org/download"
    print(f"\nStarting download to: {save_dir}")
    print(f"Target: {target_count} .pdb files")
    
    success_count = 0
    
    print(f"\n{'Status':<10} | {'PDB ID':<8} | {'Title (truncated)'}")
    print("-" * 80)

    for item in structures:
        if success_count >= target_count:
            break
            
        pdb_id = item['id']
        title = item['title']
        title_trunc = title[:50] + "..." if len(title) > 50 else title
        
        filename = f"{pdb_id}.pdb"
        file_path = os.path.join(save_dir, filename)
        download_url = f"{base_url}/{filename}"
        
        try:
            response = requests.get(download_url)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                success_count += 1
                print(f"{'SAVED':<10} | {pdb_id:<8} | {title_trunc}")
                
            elif response.status_code == 404:
                # This is expected for large structures that don't have a .pdb file
                print(f"{'SKIPPED':<10} | {pdb_id:<8} | No .pdb format available")
            else:
                print(f"{'ERROR':<10} | {pdb_id:<8} | HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"{'FAILED':<10} | {pdb_id:<8} | {e}")

    print(f"\nProcess complete. Downloaded {success_count} / {target_count} files.")

if __name__ == "__main__":
    # Settings
    TARGET_DOWNLOADS = 100
    # Fetch a larger buffer (i.e. 50) to account for skipped files; if files are not pdb -> cif
    SEARCH_BUFFER = 300 
    
    print(f"Fetching latest {SEARCH_BUFFER} RNA candidates from PDB to find {TARGET_DOWNLOADS} valid .pdb files...\n")
    
    candidates = get_latest_rna_structures(limit=SEARCH_BUFFER)
    
    if candidates:
        target_directory = os.path.join("data", "pdb_training")
        download_only_pdb_structures(candidates, target_directory, target_count=TARGET_DOWNLOADS)
    else:
        print("No results found in search.")