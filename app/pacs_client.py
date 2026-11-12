import requests
from typing import Dict, Any, List

class PACSClient:
    def __init__(self, pacs_base_url: str, auth_token: str = None):
        self.base_url = pacs_base_url.rstrip("/")
        self.headers = {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

    def query_worklist(self, patient_id: str = None) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/studies"
        params = {}
        if patient_id:
            params["PatientID"] = patient_id

        response = requests.get(url, headers=self.headers, params=params, timeout=10)
        if response.status_code != 200:
            raise Exception(f"PACS Query Failed [{response.status_code}]: {response.text}")
        return response.json()

    def fetch_dicom_instance(self, study_instance_uid: str, series_instance_uid: str, sop_instance_uid: str) -> bytes:
        url = f"{self.base_url}/studies/{study_instance_uid}/series/{series_instance_uid}/instances/{sop_instance_uid}"
        headers = {**self.headers, "Accept": "application/dicom"}

        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            raise Exception(f"WADO-RS Retrieval Failed [{response.status_code}]: {response.text}")
        return response.content
