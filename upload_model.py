import os
import logging
import tarfile
import boto3
from botocore.client import Config
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError # Import ClientError
from dotenv import load_dotenv

load_dotenv()

def upload_file(file_path, bucket, folder, access_key_id=None, secret_access_key=None):
    """Upload a file to an S3 bucket

    :param file_path: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_path is used
    :param access_key_id: AWS access key ID
    :param secret_access_key: AWS secret access key
    :return: True if file was uploaded, else False
    """
    # Tar the file
    tar_file_path = f"{file_path}.tar"
    with tarfile.open(tar_file_path, "w") as tar:
        tar.add(file_path)

    # If S3 object_name was not specified, use tar_file_path
    object_name = os.path.join(folder, os.path.basename(tar_file_path))

    print(object_name)

    # Configure the S3 client
    s3_client = boto3.client('s3',
        endpoint_url='https://71b683c614d194da725e5ba6295d8b07.r2.cloudflarestorage.com',
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(signature_version='s3v4'),
    )    

    # Configure transfer settings
    MB = 1024 ** 2
    transfer_config = TransferConfig(multipart_threshold=8 * MB, max_concurrency=10,
                                     multipart_chunksize=8 * MB, use_threads=True)

    # Upload the tarred file
    try:
        s3_client.upload_file(tar_file_path, bucket, object_name, Config=transfer_config)
        logging.info(f"Upload complete: {object_name}")
        return True
    except ClientError as e:
        logging.error(e)
        return False
    finally:
        # Remove the temporary tarred file
        os.remove(tar_file_path)

if __name__ == "__main__":
    file_path = "ComfyUI/models/unet/flux1-dev.safetensors"# = input("Enter file path: ")
    bucket = "headshotpro-public-models" #input("Enter bucket name: ")
    # object_name = input("Enter object name (leave blank to use file name): ") or None
    folder = "unet"#input("Enter AWS access key ID: ")
    access_key_id = os.environ.get("CLOUDFLARE_ACCESS_KEY")
    secret_access_key = os.environ.get("CLOUDFLARE_SECRET_KEY")

    upload_file(file_path, bucket, folder, access_key_id, secret_access_key)