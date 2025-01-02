
install:
	@echo "Is your env activated? [y/N] " && read ans && [ $${ans:-N} = y ]
	pip install -r requirements.txt
	pre-commit install

