import os
import json
import pg8000
from datetime import datetime
from decimal import Decimal


DB_HOST = os.environ["DB_HOST"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_PORT = int(os.environ.get("DB_PORT", "5432"))


def get_connection():
    return pg8000.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )


def lambda_handler(event, context):
    qs = event.get("queryStringParameters") or {}

    # Read request_id
    request_id = qs.get("request_id")

    # If no request_id, use limit to query all
    limit = 20
    if not request_id:
        try:
            if "limit" in qs and qs["limit"] is not None:
                limit = int(qs["limit"])
            if limit <= 0:
                limit = 20
            if limit > 100:
                limit = 100
        except:
            limit = 20

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # If there is request_id, query single record
        if request_id:
            query = """
                SELECT id,
                       image_name,
                       predicted_label,
                       confidence,
                       created_at,
                       request_id,
                       infer_latency_ms
                FROM predictions
                WHERE request_id = %s;
            """
            cursor.execute(query, (request_id,))
        else:
            # If no request_id, query multiple records
            query = """
                SELECT id,
                       image_name,
                       predicted_label,
                       confidence,
                       created_at,
                       request_id, 
                       infer_latency_ms
                FROM predictions
                ORDER BY created_at DESC
                LIMIT %s;
            """
            cursor.execute(query, (limit,))

        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]

        results = []
        for row in rows:
            record = {}
            for col_name, value in zip(columns, row):
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, Decimal):
                    value = float(value)
                record[col_name] = value
            results.append(record)

        cursor.close()
        conn.close()

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(results)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }
