network_config = {
    'CN1': {
        'stations': ['LZB', 'PGC', 'SNB', 'NLLB'],
        'channels': ['BHN', 'BHE', 'BHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    'CN2': {
        'stations': ['YOUB', 'PFB'],
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    'CN3': {
        'stations': ['VGZ'],
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{station}.CN.{year}.{julian_day}'
    #},
    #'CN4': {
    #    'stations': ['GOBB'],
    #    'channels': ['EHZ'],
    #    'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    'PB': {
        'stations': ['B001', 'B009', 'B010', 'B011', 'B926'],
        'channels': ['EH1', 'EH2', 'EHZ'],
        'filename_pattern': '{station}.PB.{year}.{julian_day}'
    },
    'C8': {
        'stations': ['BPCB'],
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.C8.{station}..{channel}.mseed'
    },
    'PO': {
        'stations': [],
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.PO.{station}..{channel}.mseed'
    }
}

