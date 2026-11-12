from celery import Celery
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("mammogram_workers", broker=REDIS_URL, backend=REDIS_URL)

@celery_app.task(name="workers.celery_worker.process_dbt_volume_task")
def process_dbt_volume_task(study_uid: str, series_uid: str):
    """Background worker task for processing 3D Digital Breast Tomosynthesis (DBT) volumes."""
    print(f"Starting async processing for study {study_uid}, series {series_uid}...")
    return {"status": "completed", "study_uid": study_uid}
