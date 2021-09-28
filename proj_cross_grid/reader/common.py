import datetime

__all__ = [
    'sessions',
    'bartype',
]

SESSION_1_START = datetime.time(9, 30)
SESSION_1_END = datetime.time(11, 30)
SESSION_2_START = datetime.time(13, 0)
SESSION_2_END = datetime.time(15, 0)

sessions = [
    (SESSION_1_START, SESSION_1_END),
    (SESSION_2_START, SESSION_2_END),
]

bartype = [
    ('date', 'i4'),
    ('time', 'i4'),
    ('symbol', '<S8'),
    ('preclose', 'f8'),
    ('open', 'f8'),
    ('high', 'f8'),
    ('low', 'f8'),
    ('close', 'f8'),
    ('volume', 'i8'),
    ('value', 'i8')
]