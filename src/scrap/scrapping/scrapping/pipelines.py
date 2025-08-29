import psycopg2

class ScrappingPipeline:

    def open_spider(self, spider):
        self.conn = psycopg2.connect(
            dbname="real",
            user="postgres",
            password="2024",
            host="localhost",
            port="5432",
            options='-c client_encoding=UTF8'  # ensure UTF-8 for Arabic/French
        )
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            
            CREATE TABLE IF NOT EXISTS try(
                id SERIAL PRIMARY KEY,
                title TEXT,
                price INTEGER,
                location TEXT,
                description TEXT,
                surface TEXT,
                bedrooms TEXT,
                bathrooms TEXT,
                published_time TEXT,
                image_url TEXT,
                link TEXT UNIQUE,
                type TEXT,
                property_category TEXT,
                features TEXT,
                rooms TEXT
            );
        """)
        self.conn.commit()

    def close_spider(self, spider):
        self.cursor.close()
        self.conn.close()

    def process_item(self, item, spider):
        try:
            # Nettoyage et conversion du prix
            price_raw = item.get('price', '')
            price_clean = None
            if price_raw:
                # supprimer les espaces insécables et les autres espaces
                price_clean = int(price_raw.replace('\u202f', '').replace(' ', '').replace('Dhs', '').strip())

            self.cursor.execute("""
                INSERT INTO try (
                    title, price, location, description, surface, bedrooms, bathrooms,
                    published_time, image_url, link, type, property_category, features, rooms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (link) DO NOTHING
            """, (
                item.get('title', ''),
                price_clean,
                item.get('location', ''),
                item.get('description', ''),
                item.get('surface', ''),
                item.get('bedrooms', ''),
                item.get('bathrooms', ''),
                item.get('published_time', ''),
                item.get('image_url', ''),
                item.get('link', ''),
                item.get('type', ''),
                item.get('property_category', ''),
                item.get('features', ''),
                item.get('rooms', '')
            ))
            self.conn.commit()
        except Exception as e:
            spider.logger.error(f"Database insert failed: {e}")
            self.conn.rollback()  # reset transaction state so other items can continue
        return item
