import scrapy
from scrapping.items import AvitoItem

class AvitoSpider(scrapy.Spider):
    name = "avito"
    allowed_domains = ["avito.ma"]

    base_urls = [
       
        ("https://www.avito.ma/fr/maroc/appartements-%C3%A0_louer", "For Rent")
    ]

    def start_requests(self):
        for base_url, listing_type in self.base_urls:
            for page in range(1, 300):
                url = f"{base_url}?o={page}"
                yield scrapy.Request(url=url, callback=self.parse, meta={'listing_type': listing_type})

    def parse(self, response):
        for ad in response.css('a.sc-1jge648-0.jZXrfL'):
            link = response.urljoin(ad.css('::attr(href)').get())
            meta_data = {
                'link': link,
                'title': ad.css('p[title]::attr(title)').get(default='').strip(),
                'price': ad.css('p.dJAfqm span::text').get(default='').strip(),
                'location': ad.css('div.kclCPb > p::text').get(default='').strip(),
                'published_time': ad.css('p.layWaX::text').get(default='').strip(),
                'image_url': ad.css('img.sc-bsm2tm-3::attr(src)').get(default=''),
                'type': response.meta['listing_type']
            }

            # chambres / sdb / surface
            spans = ad.css('div.sc-b57yxx-2 span')
            for s in spans:
                label = s.css('div::attr(title)').get()
                value = s.css('span::text').get()
                if label == 'Chambres':
                    meta_data['bedrooms'] = value
                elif label == 'Salle de bain':
                    meta_data['bathrooms'] = value
                elif label == 'Surface totale':
                    meta_data['surface'] = value

            yield scrapy.Request(url=link, callback=self.parse_detail, meta=meta_data)

    def parse_detail(self, response):
        item = AvitoItem()

        item['link'] = response.meta['link']
        item['title'] = response.meta['title']
        item['price'] = response.meta['price']
        item['location'] = response.meta['location']
        item['published_time'] = response.meta['published_time']
        item['image_url'] = response.meta['image_url']
        item['type'] = response.meta['type']
        item['surface'] = response.meta.get('surface', '')
        item['bedrooms'] = response.meta.get('bedrooms', '')
        item['bathrooms'] = response.meta.get('bathrooms', '')
        desc_parts = response.xpath('//h2[text()="Description"]/following-sibling::div//text()').getall()
        item['description'] = ' '.join([d.strip() for d in desc_parts if d.strip()])
        item['property_category'] = "Apartment"
        item['rooms'] = item['bedrooms']

        # ✅ Extract features
        features = response.css('div.sc-19cngu6-2 span::text').getall()
        item['features'] = ', '.join([f.strip() for f in features])

        yield item
