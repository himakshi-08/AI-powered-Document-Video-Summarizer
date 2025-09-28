from newspaper import Article

def scrape_article(url):
    """
    Scrapes an article from a given URL and returns its details.

    Args:
        url: The URL of the article to scrape.

    Returns:
        A dictionary containing the article's title, authors, publish date, and text.
        Returns None if scraping fails.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()

        return {
            'title': article.title,
            'authors': article.authors,
            'publish_date': article.publish_date,
            'text': article.text
        }
    except Exception as e:
        print(f"Error scraping article from {url}: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    test_url = "https://www.bbc.com/news/world-us-canada-67467270"  # Replace with a real article URL
    article_data = scrape_article(test_url)

    if article_data:
        print("Scraped Article Details:")
        print(f"Title: {article_data['title']}")
        print(f"Authors: {article_data['authors']}")
        print(f"Publish Date: {article_data['publish_date']}")
        print(f"Text (first 500 chars): {article_data['text'][:500]}...")
    else:
        print("Failed to scrape article.")