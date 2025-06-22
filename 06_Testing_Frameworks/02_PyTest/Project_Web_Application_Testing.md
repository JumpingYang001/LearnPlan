# Project: Web Application Testing

## Description
Create tests for a web application using PyTest and Selenium, with fixtures and page object patterns.

## Example
```python
import pytest
from selenium import webdriver

@pytest.fixture
def browser():
    driver = webdriver.Chrome()
    yield driver
    driver.quit()

def test_homepage(browser):
    browser.get("https://example.com")
    assert "Example Domain" in browser.title
```
