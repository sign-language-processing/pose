lint:
	cd src/python && yapf -i -r .
	cd src/python && isort .
	cd src/python && pydoclint .

test:
	cd src/python && pytest .