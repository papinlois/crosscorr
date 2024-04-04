network_config = {
    '1CN': {
        'stations': ['LZB', 'PGC', 'SNB', 'NLLB'], #
        'channels': ['BHN', 'BHE', 'BHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    '2CN': {
        'stations': ['YOUB', 'PFB'], #
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    '3CN': {
        'stations': ['VGZ'], #
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{station}.CN.{year}.{julian_day}'
    },
    '4CN': {
        'stations': ['GOBB'], #
        'channels': ['EHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    '1C8': {
        'stations': ['BPCB', 'MGCB', 'JRBC', 'GLBC', 'SHVB', 'TWBB'], #
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.C8.{station}..{channel}.mseed'
    },
    '2C8': {
        'stations': ['SHDB'], #
        'channels': ['BHN', 'BHE', 'BHZ'],
        'filename_pattern': '{date}.C8.{station}..{channel}.mseed'
    },
    'PO': {
        'stations': ['SILB', 'SSIB', 'TSJB', 'TWKB', 'KLNB'], #
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.PO.{station}..{channel}.mseed'
    },
    'PB': {
        'stations': ['B001', 'B009', 'B010', 'B011', 'B926'], #
        'channels': ['EH1', 'EH2', 'EHZ'],
        'filename_pattern': '{station}.PB.{year}.{julian_day}'
    }
}
