# **Model Overview**

- We include three models, each folder contains a poetry-managed environment. 

## **Setting Up the Poetry Environment**
We use **Poetry** for dependency management. Follow these steps to install and set up the environment:

### **1.1 Install Poetry (if not installed)**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
or use:

```bash 
pip install poetry
```


### **1.3 Install Dependencies**
```bash
poetry install
```
- Be in the same directory as the .toml and .lock files for this to work.
- This will install all the necessary dependencies in the poetry virtual environmnet.
