# Scraping des données

## Sources
- **Mubawab** : Selenium
- **Avito** : Scrapy

## Données collectées
- Titre,Description,Prix, surface, chambres,salle de bains, localisation, type de bien,catégorie, images_url,link

## Notes sur les outils
- **Selenium** : automatisation de navigation web pour récupérer des pages dynamiques  
  Documentation : [Selenium](https://www.selenium.dev/documentation/)
- **Scrapy** : framework de scraping pour crawler les annonces  
  Documentation : [Scrapy](https://docs.scrapy.org/en/latest/)

## Remarques
- Les scripts gèrent les exceptions et pages manquantes
- Données brutes stockées dans `data/raw/`
