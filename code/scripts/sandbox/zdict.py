class zdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['z'] = self['x']

class zdict2(dict):
    def __getitem__(self, item):
        if item == 'z':
            return super().__getitem__('x')
        else:
            return super().__getitem__(item)


mydict = {
    'a1': zdict(
        x=1,
        y=2
    ),
    'a2':zdict(
        x=3,
        y=4
    ),
}

print(mydict['a1']['z'], mydict['a1']['x'])
print(mydict['a2']['z'], mydict['a2']['x'])



mydict2 = {
    'a1': zdict2(
        x=1,
        y=2
    ),
    'a2':zdict2(
        x=3,
        y=4
    ),
}

print(mydict2['a1']['z'], mydict2['a1']['x'])
print(mydict2['a2']['z'], mydict2['a2']['x'])


mydict3 = {
    'a1': dict(
        x=1,
        y=2
    ),
    'a2': dict(
        x=3,
        y=4
    ),
}

