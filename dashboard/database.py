import os
import pandas as pd
import random
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "churn_crm"
COLLECTION_NAME = "customers"

def get_collection():
    """Get the MongoDB collection."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def init_db():
    """Initialize the MongoDB collection (create indexes)."""
    try:
        collection = get_collection()
        # Create unique index on customerID
        collection.create_index("customerID", unique=True)
        # Create index on risk_level for faster querying
        collection.create_index("risk_level")
        print("MongoDB initialized and indexes created.")
    except Exception as e:
        print(f"Error initializing MongoDB: {e}")

def seed_database(csv_path, n_samples=200):
    """Seed the database with sample data if empty."""
    try:
        collection = get_collection()
        
        # Check if empty
        if collection.count_documents({}) == 0 and os.path.exists(csv_path):
            try:
                # Load and sample data
                try:
                    df = pd.read_csv(csv_path)
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_path, encoding='latin1')
                
                # Handle TotalCharges
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
                
                # Drop Churn column if exists
                if 'Churn' in df.columns:
                    df = df.drop(columns=['Churn'])
                
                # Sample
                if len(df) > n_samples:
                    df = df.sample(n=n_samples, random_state=42)
                
                # Generate random names
                first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen"]
                last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
                
                df['customer_name'] = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(len(df))]
                df['avatar_seed'] = df['customerID']

                # Add missing columns
                df['churn_probability'] = None
                df['churn_prediction'] = None
                df['risk_level'] = 'Unknown'
                df['follow_up_status'] = 'Pending'
                df['notes'] = ''
                df['contact_email'] = df['customerID'].apply(lambda x: f"{x.lower()}@example.com")
                df['contact_phone'] = df['customerID'].apply(lambda x: f"555-{random.randint(100,999)}-{random.randint(1000,9999)}")
                
                # Convert DataFrame to list of dicts for insertion
                records = df.to_dict('records')
                collection.insert_many(records)
                print(f"Seeded database with {len(records)} records.")
            except Exception as e:
                print(f"Error processing data for seed: {e}")
    except Exception as e:
        print(f"Error connecting to MongoDB for seeding: {e}")

def load_customers():
    """Load all customers from the database."""
    collection = get_collection()
    try:
        cursor = collection.find({}, {'_id': 0}) # Exclude _id field
        df = pd.DataFrame(list(cursor))
        return df
    except Exception as e:
        print(f"Error loading customers: {e}")
        return pd.DataFrame()

def save_predictions(df):
    """Save or update customer predictions in the database."""
    collection = get_collection()
    
    # Generate mock contact info if not exists (for new records in df)
    if 'contact_email' not in df.columns:
        df['contact_email'] = df['customerID'].apply(lambda x: f"{x.lower()}@example.com")
    if 'contact_phone' not in df.columns:
        df['contact_phone'] = df['customerID'].apply(lambda x: f"555-{random.randint(100,999)}-{random.randint(1000,9999)}")
    
    # Ensure status columns exist
    if 'follow_up_status' not in df.columns:
        df['follow_up_status'] = 'Pending'
    if 'notes' not in df.columns:
        df['notes'] = ''

    # Prepare bulk operations
    operations = []
    for _, row in df.iterrows():
        # We use $set to update fields. 
        # Note: This will overwrite existing fields with values from df.
        # Since df comes from load_customers() + updates, this is generally safe.
        
        data = row.to_dict()
        customer_id = data.pop('customerID') # Remove ID from data to avoid modifying immutable _id if we were using it, but here we use customerID as filter
        
        operations.append(
            UpdateOne(
                {"customerID": customer_id},
                {"$set": data},
                upsert=True
            )
        )
    
    if operations:
        try:
            collection.bulk_write(operations)
        except Exception as e:
            print(f"Error saving predictions: {e}")

def update_customer_status(customer_id, status, notes):
    """Update the follow-up status and notes for a customer."""
    collection = get_collection()
    try:
        collection.update_one(
            {"customerID": customer_id},
            {"$set": {"follow_up_status": status, "notes": notes}}
        )
    except Exception as e:
        print(f"Error updating status: {e}")

def get_high_risk_customers():
    """Get customers with High risk level."""
    collection = get_collection()
    try:
        cursor = collection.find({"risk_level": "High"}, {'_id': 0}).sort("churn_probability", -1)
        df = pd.DataFrame(list(cursor))
        return df
    except Exception as e:
        print(f"Error getting high risk customers: {e}")
        return pd.DataFrame()
