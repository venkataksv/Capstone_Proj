import boto3
import json
from datetime import datetime

def upload_to_s3(bucket_name, object_name, data):
    # Initialize S3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id='',
        aws_secret_access_key='',
        region_name='us-east-2'  # Change to your bucket's region
    )
    
    try:
        # Upload the file
        response = s3.put_object(
            Bucket=bucket_name,
            Key=object_name,
            Body=json.dumps(data),
            ContentType='application/json'
        )
        print(f"Uploaded {object_name} to bucket {bucket_name}")
        return response
    except Exception as e:
        print(f"Error uploading to S3: {e}")

# Example usage
if __name__ == "__main__":
    detection_result = {
        "label": "Apple",
        "confidence": 0.95,
        "timestamp": datetime.now().isoformat()
    }
    bucket_name = "fruitsense-detection-results-bucket"
    object_name = f"detection_results/{detection_result['timestamp']}.json"

    upload_to_s3(bucket_name, object_name, detection_result)