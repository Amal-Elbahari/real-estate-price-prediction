import psycopg2
import re
from datetime import datetime
from typing import Optional, List, Tuple
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class RealEstateDataCleaner:
    def __init__(self):
        self.target_features = ["terrace", "garage", "elevator", "concierge", "pool", "security", "garden"]
        self.db_params = {
            "dbname": "real_estate",
            "user": "postgres",
            "password": "2024",
            "host": "localhost",
            "port": "5432"
        }

    def clean_price(self, price_str: str) -> Optional[float]:
        """Clean and validate price value"""
        if not price_str:
            return None
            
        try:
            price = float(re.sub(r"[^\d.]", "", str(price_str)))
            if not (10000 <= price <= 100000000):
                return None
            return price
        except (ValueError, TypeError):
            return None

    def clean_surface(self, surface_str: str) -> Optional[float]:
        """Clean and validate surface area"""
        if not surface_str:
            return None
            
        try:
            match = re.search(r"(\d+\.?\d*)", str(surface_str).replace(",", "."))
            if not match:
                return None
            surface = float(match.group(1))
            if not (10 <= surface <= 2000):
                return None
            return surface
        except (ValueError, TypeError, AttributeError):
            return None

    def clean_integer(self, value: str, min_val: int = 0, max_val: int = 20) -> Optional[int]:
        """Clean and validate integer fields"""
        if not value:
            return None
            
        try:
            num = int(re.sub(r"[^\d]", "", str(value)))
            if not (min_val <= num <= max_val):
                return None
            return num
        except (ValueError, TypeError):
            return None

    def clean_text(self, text: str) -> Optional[str]:
        """Clean and validate text fields"""
        if not text:
            return None
        return re.sub(r'\s+', ' ', str(text).strip())

    def parse_features(self, features_str: str) -> List[bool]:
        """Parse property features into binary flags"""
        if not features_str:
            return [False] * len(self.target_features)
        try:
            features = [f.strip().lower() for f in str(features_str).split(",")]
            return [any(feat in f for f in features) for feat in self.target_features]
        except (AttributeError, TypeError):
            return [False] * len(self.target_features)

    def process_data(self):
        """Main data processing workflow"""
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cursor:
                    # Create table with constraints
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS clean_sall(
                            id SERIAL PRIMARY KEY,
                            title TEXT NOT NULL,
                            price FLOAT NOT NULL CHECK (price > 0),
                            location TEXT NOT NULL,
                            description TEXT,
                            surface FLOAT NOT NULL CHECK (surface > 0),
                            rooms INT CHECK (rooms BETWEEN 0 AND 20),
                            bedrooms INT CHECK (bedrooms BETWEEN 0 AND 20),
                            bathrooms INT CHECK (bathrooms BETWEEN 0 AND 10),
                            type TEXT,
                            property_category TEXT,
                            link TEXT,
                            image_url TEXT,
                            published_time TEXT,
                            terrace BOOLEAN DEFAULT FALSE,
                            garage BOOLEAN DEFAULT FALSE,
                            elevator BOOLEAN DEFAULT FALSE,
                            concierge BOOLEAN DEFAULT FALSE,
                            pool BOOLEAN DEFAULT FALSE,
                            security BOOLEAN DEFAULT FALSE,
                            garden BOOLEAN DEFAULT FALSE,
                            insert_date TIMESTAMP NOT NULL
                        );
                    """)
                    conn.commit()

                    # Load raw data
                    cursor.execute("""
                        SELECT title, price, location, description, surface,
                               rooms, bedrooms, bathrooms, type,
                               property_category, link, image_url, published_time, features
                        FROM mubawab_sall
                    """)
                    rows = cursor.fetchall()

                    cleaned_rows = []
                    skipped_count = 0

                    for row in rows:
                        # Clean numeric fields first
                        price = self.clean_price(row[1])
                        surface = self.clean_surface(row[4])
                        rooms = self.clean_integer(row[5])
                        bedrooms = self.clean_integer(row[6])
                        bathrooms = self.clean_integer(row[7])

                        # Only proceed if we have valid numeric fields
                        if None in [price, surface]:
                            skipped_count += 1
                            continue

                        # Clean text fields
                        title = self.clean_text(row[0])
                        location = self.clean_text(row[2])
                        description = self.clean_text(row[3])
                        prop_type = self.clean_text(row[8])
                        prop_category = self.clean_text(row[9])
                        link = self.clean_text(row[10])
                        image_url = self.clean_text(row[11])
                        published_time = self.clean_text(row[12])
                        features = self.parse_features(row[13])

                        # Validate business rules
                        if rooms and surface / rooms < 5:
                            skipped_count += 1
                            continue

                        if not (500 <= price / surface <= 50000):
                            skipped_count += 1
                            continue

                        # Add to cleaned data
                        cleaned_rows.append((
                            title, price, location, description, surface,
                            rooms, bedrooms, bathrooms, prop_type,
                            prop_category, link, image_url, published_time,
                            *features,
                            datetime.now()
                        ))

                    # Batch insert
                    if cleaned_rows:
                        insert_query = """
                            INSERT INTO clean_sall(
                                title, price, location, description, surface,
                                rooms, bedrooms, bathrooms, type,
                                property_category, link, image_url, published_time,
                                terrace, garage, elevator, concierge, pool, security, garden,
                                insert_date
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        # Insert in batches of 1000
                        batch_size = 1000
                        for i in range(0, len(cleaned_rows), batch_size):
                            cursor.executemany(insert_query, cleaned_rows[i:i+batch_size])
                            conn.commit()

                        logging.info(f"✅ Inserted {len(cleaned_rows)} clean records")
                        logging.info(f"⚠️ Skipped {skipped_count} invalid records")
                    else:
                        logging.warning("⚠️ No valid records to insert")

        except psycopg2.Error as e:
            logging.error(f"Database error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    cleaner = RealEstateDataCleaner()
    cleaner.process_data()