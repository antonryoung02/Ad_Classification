
import pandas as pd
import requests
import json
import os

class Openverse:
    def register(self):
        url = 'https://api.openverse.org/v1/auth_tokens/register/'
        data = {
            "name": "Sports Ad Classifier",
            "description": "Gathering sports image data for ML Classification task",
            "email": "email@gmail.com"
        }
        headers = {
            'Content-Type': 'application/json',
        }
        response = requests.post(url=url, json=data, headers=headers)
        if response.status_code == 201:
            response_data = response.json()
            client_id = response_data.get('client_id')
            client_secret = response_data.get('client_secret')
            
            credentials = {
                'client_id': client_id,
                'client_secret': client_secret
            }
            with open('openverse_credentials.json', 'w') as file:
                json.dump(credentials, file)
        else:
            print(f'Error: {response.status_code} - {response.text}')
            
    def verify_email(self):
        url = 'https://api.openverse.org/v1/auth_tokens/verify/25Ql1AgGM9cq55OXW_vOPEuisUSIys2y5v5o2B5RckRQYxXJqBCsk-g5BKun2d6Qtcl2a3zyix3mQmEemPDKjw/'
        response = requests.get(url=url)
        response_data = response.json()
        if response.status_code == 200:
            response_data = response.json()
            print(f"Email verification successful: {response_data}")
        else:
            print(f"Error {response.status_code}: {response.text}")

    def get_access_token(self):
        with open('./data_processing/openverse_credentials.json', 'r') as file:
            data = json.load(file)
            client_id = data.get('client_id')
            client_secret = data.get('client_secret')
            print(f"Client ID: {client_id}")
            print(f"Client Secret: {client_secret}")
            data = {
                "client_id":client_id,
                "client_secret":client_secret,
                "grant_type":"client_credentials",
            }
            url = "https://api.openverse.org/v1/auth_tokens/token/"
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            response = requests.post(url=url, data=data, headers=headers)
            if response.status_code == 200:
                response_data = response.json()
                access_token = response_data.get('access_token')
                print("Access token received:", access_token)
                return access_token
            else:
                print(f'Error {response.status_code}: {response.text}')
                return None
            
    def get_images(self, openverse_query, openverse_access_token):
        """
        Expects dictionary of params as decined in the API documentation 
        https://api.openverse.org/v1/#tag/images/operation/images_search
        """
        url = "https://api.openverse.org/v1/images/"
        headers = {
            'Authorization': f'Bearer {openverse_access_token}'
        }
        query_params = openverse_query.get_query_params()
        response = requests.get(url=url, params=query_params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"number of images: {data['result_count']}")
            return data
        else:
            print(f'Error {response.status_code}: {response.text}')
            return None
        
    def save_images_to_directory(self, response, save_dir):
        for image in response["results"]:
            self.download_image(image, save_dir)
            
    def download_image(self, image, save_dir):
        image_url = image['url']
        filetype = image['filetype'] # Use 'jpg' as default if filetype is None
        if filetype is None:
            filetype = "jpg"
        
        response = requests.get(image_url)
        if response.status_code == 200:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            image_name = os.path.basename(image['id']) + '.' + filetype
            image_path = os.path.join(save_dir, image_name)
            
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded {image_name}")
        else:
            print(f"Failed to download {image_url}: {response.status_code}")

        
        

class OpenverseQuery:
    def __init__(self, query_str, query_params):
        self.query_params = {"q":query_str,'license': 'by,by-sa,cc0,pdm', 'license_type': 'commercial,modification'}
        self.accepted_params = {
            "page", "page_size", "q", "source", "excluded_source", "tags", "title",
            "creator", "unstable__collection", "unstable__tag", "license", "license_type",
            "filter_dead", "extension", "mature", "unstable__sort_by", "unstable__sort_dir",
            "unstable__authority", "unstable__authority_boost", "unstable__include_sensitive_results",
            "category", "aspect_ratio", "size"
        }
        for key, val in query_params.items():
            if key in self.accepted_params:
                self.query_params[key] = val
            else:
                print(f"query key {key} not in the query's accepted parameter set")
    
    def set_query_value(self, key, val):
        if key in self.accepted_params:
            self.query_params[key] = val
        else:
            print(f"query key {key} not in the query's accepted parameter set")
                
    def get_query_params(self):
        return self.query_params
                