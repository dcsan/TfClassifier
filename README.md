# TfClassifier

Example package using TensorFlowJS to provide a classifier

## tests
Makefile has basic tasks

try `make test` to see an example run using [test data](src/data/inputs/test.csv)

Have a look at the [TestRunner](src/TestRunner.ts) for example usage.

## training
To train the classifier, you need to pass an array of phrases:

(csv example)

```csv
tag,text
READ,   I need to read more
GYM,    Go to the gym
READ,   do more reading
```

You can (and should!) have multiple training phrases per tag.

Load the data, and then train the classifier:

```ts
    await testModel.loadCsvInputs('./data/inputs/train.csv')
    await testModel.trainModel({ useCache: useCache })
```

then you can see the best match for new phrases:

```ts
const matches: IMatch[] | undefined =
  await testModel.classify('I like reading books', { expand: true })
```

The returned results will be a list of matches with `tag`

If you passed `expand: true` you'll also get all the training samples which had that tag

```js
input: 'I need to read more'

matches: [
  [
    confidence: 0.3560619056224823,
    tag: 'READ',
    pct: 36,
    sources: [
      { tag: 'READ', text: 'Read more books' },
      { tag: 'READ', text: "Read a children's book" },
      { tag: 'READ', text: 'Read the news in English Language' },
      { tag: 'READ', text: 'Finish a new book' },
      { tag: 'READ', text: 'Read a book every day' },
      { tag: 'READ', text: 'finish reading Content Marketing Part 1' }
    ]
  ],
  ...
]
```


tag: a name for that 'class' of the classifier
text: example phrase


