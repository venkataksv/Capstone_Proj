import boto3
import json
from datetime import datetime

def list_s3_objects(bucket_name, prefix):
    # Initialize S3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id='',
        aws_secret_access_key='',
        region_name='us-east-2'
    )
    
    try:
        # List objects
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                print(f"File: {obj['Key']}")
        else:
            print("No files found.")
    except Exception as e:
        print(f"Error listing S3 objects: {e}")

# Example usage
if __name__ == "__main__":
    bucket_name = "fruitsense-detection-results-bucket"
    list_s3_objects(bucket_name, "detection_results/")
