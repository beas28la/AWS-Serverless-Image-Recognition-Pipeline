import json
import os
import uuid
import boto3
import base64

s3 = boto3.client("s3")
UPLOAD_BUCKET = os.environ["UPLOAD_BUCKET"]

def lambda_handler(event, context):
    print("EVENT:", json.dumps({
        "keys": list(event.keys()),
        "isBase64Encoded": event.get("isBase64Encoded"),
        "headers": event.get("headers")
    }))

    # 1. Get headers & content-type
    headers = event.get("headers") or {}
    # Sometimes it's 'content-type', sometimes it's 'Content-Type'
    content_type = (headers.get("content-type")
                    or headers.get("Content-Type")
                    or "application/octet-stream")

    body = event.get("body", "")

    # 2. Check if it's binary (API Gateway will set isBase64Encoded = true)
    image_bytes = None
    ext = ".bin"

    if event.get("isBase64Encoded"):
        # binary upload path: body is base64
        try:
            image_bytes = base64.b64decode(body)
        except Exception as e:
            print("Failed to decode base64 body:", e)
            return _error(400, "invalid base64 binary body")

        # Guess the extension based on content-type
        if "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        elif "png" in content_type:
            ext = ".png"
        else:
            ext = ".bin"

    else:
        # Compatible with previous JSON + image_base64 writing (in case you want to use it later)
        try:
            parsed = json.loads(body) if body else {}
        except Exception as e:
            print("Failed to parse JSON body:", e)
            parsed = {}

        image_b64 = parsed.get("image_base64")
        if not image_b64:
            return _error(400, "no binary body and no image_base64 field")

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            print("Failed to decode image_base64:", e)
            return _error(400, "invalid image_base64 field")

        ext = parsed.get("extension", "jpg")
        if not ext.startswith("."):
            ext = "." + ext

    # 3. Generate request_id and S3 key
    request_id = str(uuid.uuid4())
    s3_key = f"{request_id}/image{ext}"

    # 4. Write to S3
    s3.put_object(
        Bucket=UPLOAD_BUCKET,
        Key=s3_key,
        Body=image_bytes,
        ContentType=content_type
    )

    # 5. Return request_id, later /results can use this to lookup
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "request_id": request_id,
            "bucket": UPLOAD_BUCKET,
            "s3_key": s3_key,
            "content_type": content_type
        })
    }


def _error(status, msg):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": msg})
    }
