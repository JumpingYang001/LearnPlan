# CI/CD for Machine Learning

## Topics
- MLOps principles
- Model versioning and registry
- Automated testing for ML models
- CI/CD pipelines for ML systems

### Example: GitHub Actions for ML CI
```yaml
name: ML CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest
```
