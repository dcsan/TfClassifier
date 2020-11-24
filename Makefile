clean:
	rm -rf dist/*

test:
	npm run test

bump:
	npm version patch

build: clean
	npm run build

publish: clean test bump
	npm publish --access public

# no test
quickpub: clean bump
	npm publish --access public
