# fine-grained-representation

The extraction of fine - grained representations (at the phoneme level) is mainly achieved with the help of the emotional dimension model. For specific model usage, please refer to: [w2v2-how-to](https://github.com/audeering/w2v2-how-to)

For LJSpeech, you can run

```
python extract_fine_feature_for_ljspeech.py
```

The directory structure of LJSpeech dataset is consistent with that provided by the official source.

For ESD, you can run

```
python extract_fine_feature.py
```

The directory structure of ESD is in MFA format.

```
-- speaker 1
	-- basename1.wav
	-- basename1.lab
	-- ...
-- speaker 2
	-- basename1.wav
	-- basename1.lab
	--...
-- ...

```

