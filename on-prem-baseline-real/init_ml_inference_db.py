# File to initialize PostgreSQL database and creates tables to to save ml model results
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database(DB_name):
    conn = None
    cursor = None
    try:
        # Connect to the default 'postgres' database
        conn = psycopg2.connect(
            database="postgres",
            user="",
            password="",
             host="localhost",
            port="5432"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Create the new database if it doesn't exist
        cursor.execute(f"CREATE DATABASE {DB_name}")
        print(f"Database '{DB_name}' created successfully.")

    except psycopg2.errors.DuplicateDatabase:
        print(f"Database '{DB_name}' already exists.")
    except psycopg2.Error as e:
        print(f"Error creating database: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Establish a new connection to the newly created db 
def create_tables(DB_name): 
    conn = None
    cursor = None
    try:
        # Connect to the newly created database
        conn = psycopg2.connect(
            dbname=DB_name,
            user="",
            password="",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()
        
        # SQL to create table to store image results 
        create_predictions_table_sql = """
        CREATE TABLE predictions (
        id SERIAL PRIMARY KEY,
        image_name VARCHAR(255) NOT NULL,
        predicted_label VARCHAR(100) NOT NULL,
        confidence FLOAT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_predictions_table_sql)

        # ----------------------------------------
        ## Add indexes to improve query performance 
        # ----------------------------------------

        # Add index for querying by predictions timestamp
        add_idx_preds_created_at_sql = """
        CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
        """
        cursor.execute(add_idx_preds_created_at_sql)
        
        # Add index for querying by prediction labels 
        add_idx_preds_label_sql = """
        CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(predicted_label);
        """
        cursor.execute(add_idx_preds_label_sql)
 
        conn.commit() 
        print("Tables created successfully (if not existing).")

    except psycopg2.Error as e:
        print(f"Error creating tables: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    DB_name = "ml_inference"
    create_database(DB_name)
    create_tables(DB_name)

