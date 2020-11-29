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

# run tests after
quickpub: clean bump
	npm publish --access public
	npm run test

# leave the .gitignore
cleanCaches:
	rm -rf src/data/modelCache/*
