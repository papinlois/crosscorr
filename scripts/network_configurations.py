network_config = {
    '1CN': {
        'stations': ['LZB','PGC', 'NLLB', 'SNB', 'VGZ'], #
        'channels': ['BHN', 'BHE', 'BHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    '2CN': {
        'stations': ['PFB', 'YOUB'], #
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    'C8': {
        'stations': ['MGCB', 'JRBC', 'PHYB', 'SHVB','LCBC', 'GLBC', 'TWBB'], # 
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.C8.{station}..{channel}.mseed'
    },
    'PO': {
        'stations': ['TSJB', 'SILB', 'SSIB', 'KLNB', 'TWKB'], # 
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.PO.{station}..{channel}.mseed'
    }
}