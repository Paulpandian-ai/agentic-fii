"""
Stock Universe - Lists of stocks from NYSE and NASDAQ for screening.

Provides curated lists of:
- S&P 500 components
- NASDAQ 100 components
- Dow Jones 30 components
- Sector-specific stocks
"""

from typing import Optional


# Sector ETFs for reference
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


# Dow Jones 30 Components
DOW_30 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
]


# NASDAQ 100 Components (Top tech-heavy stocks)
NASDAQ_100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "PEP",
    "COST", "CSCO", "ADBE", "NFLX", "AMD", "CMCSA", "INTC", "INTU", "QCOM", "TXN",
    "TMUS", "AMGN", "ISRG", "HON", "AMAT", "BKNG", "SBUX", "ADP", "GILD", "MDLZ",
    "ADI", "VRTX", "LRCX", "REGN", "PANW", "MU", "PYPL", "SNPS", "KLAC", "CDNS",
    "ASML", "ORLY", "MELI", "CSX", "MNST", "MAR", "NXPI", "CHTR", "WDAY", "CTAS",
    "ABNB", "MRVL", "PCAR", "FTNT", "KDP", "AEP", "DXCM", "ADSK", "PAYX", "CPRT",
    "ODFL", "KHC", "EXC", "ROST", "CRWD", "MRNA", "IDXX", "FAST", "CEG", "VRSK",
    "CTSH", "EA", "GEHC", "BKR", "XEL", "DLTR", "FANG", "ANSS", "DDOG", "CSGP",
    "ZS", "ILMN", "TEAM", "WBD", "BIIB", "ALGN", "EBAY", "WBA", "ENPH", "SIRI",
    "JD", "LCID", "RIVN", "ZM", "DOCU", "OKTA", "SPLK", "MTCH", "LULU", "TTD"
]


# S&P 500 - Major Components by Sector
SP500_BY_SECTOR = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "CSCO", "ADBE", "CRM", "ORCL", "ACN", "IBM",
        "INTC", "AMD", "QCOM", "TXN", "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS",
        "ADI", "MCHP", "HPQ", "HPE", "DELL", "KEYS", "MPWR", "NXPI", "ON", "SWKS",
        "FTNT", "PANW", "NOW", "INTU", "ADSK", "ANSS", "PTC", "PAYC", "PAYX", "ADP",
        "CTSH", "IT", "DXC", "EPAM", "AKAM", "FFIV", "JNPR", "NTAP", "WDC", "STX"
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDT", "SYK", "BSX", "ZBH", "EW",
        "BDX", "DXCM", "IDXX", "IQV", "A", "MTD", "WAT", "PKI", "HOLX", "TFX",
        "CVS", "CI", "ELV", "HUM", "CNC", "MOH", "HCA", "UHS", "THC", "DVA",
        "MRNA", "BIIB", "ALGN", "ZTS", "VTRS", "OGN", "CTLT", "CRL", "DGX", "LH"
    ],
    "Financials": [
        "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "SCHW", "C",
        "AXP", "BLK", "SPGI", "CME", "ICE", "MCO", "MSCI", "MMC", "AON", "AJG",
        "USB", "PNC", "TFC", "COF", "BK", "STT", "NTRS", "AMP", "TROW", "RJF",
        "PRU", "MET", "AFL", "PGR", "TRV", "ALL", "CB", "HIG", "AIG", "LNC",
        "FITB", "RF", "HBAN", "CFG", "KEY", "MTB", "ZION", "CMA", "FRC", "WAL"
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
        "ORLY", "AZO", "ROST", "DG", "DLTR", "ULTA", "BBY", "KMX", "EBAY", "ETSY",
        "MAR", "HLT", "WYNN", "LVS", "MGM", "RCL", "CCL", "NCLH", "EXPE", "ABNB",
        "GM", "F", "APTV", "BWA", "LEA", "LKQ", "GPC", "AAP", "AN", "LAD",
        "YUM", "DRI", "WING", "TXRH", "EAT", "DPZ", "WEN", "QSR", "DINE", "CAKE"
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "EL",
        "GIS", "K", "KHC", "HSY", "MKC", "SJM", "CAG", "CPB", "HRL", "TSN",
        "KR", "SYY", "WBA", "TGT", "DG", "DLTR", "BJ", "ADM", "BG", "INGR",
        "STZ", "TAP", "BF.B", "DEO", "SAM", "MNST", "KDP", "FIZZ", "CELH", "COKE",
        "CLX", "CHD", "SPB", "ENR", "CWH", "NWL", "HELE", "PRGO", "PBH", "COTY"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "PXD",
        "WMB", "KMI", "OKE", "EP", "TRGP", "LNG", "HES", "DVN", "FANG", "APA",
        "BKR", "HAL", "NOV", "FTI", "CHK", "RRC", "AR", "SWN", "CNX", "EQT",
        "MRO", "CTRA", "MTDR", "CHRD", "PR", "SM", "PDCE", "GPOR", "MGY", "VTLE"
    ],
    "Industrials": [
        "UNP", "UPS", "HON", "RTX", "CAT", "DE", "BA", "LMT", "GE", "MMM",
        "GD", "NOC", "EMR", "ITW", "ETN", "PH", "ROK", "DOV", "AME", "OTIS",
        "FDX", "CSX", "NSC", "DAL", "UAL", "AAL", "LUV", "JBLU", "RYAAY", "CHRW",
        "WM", "RSG", "WCN", "VRSK", "GWW", "FAST", "WSO", "POOL", "SNA", "TRMB",
        "IR", "XYL", "IEX", "ROP", "GNRC", "CMI", "PCAR", "PACW", "AGCO", "TT"
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "DOW", "PPG",
        "VMC", "MLM", "STLD", "CF", "MOS", "ALB", "FMC", "IFF", "CE", "EMN",
        "CTVA", "IP", "PKG", "WRK", "SEE", "SON", "AVY", "BLL", "CCK", "AMCR",
        "RPM", "AXTA", "RGLD", "GOLD", "AEM", "KGC", "WPM", "FNV", "AU", "AG"
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "PCG",
        "WEC", "ES", "AWK", "DTE", "PPL", "CMS", "EIX", "AES", "FE", "ETR",
        "EVRG", "ATO", "NI", "CNP", "PNW", "OGE", "NRG", "VST", "AEE", "LNT"
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "WELL", "SPG", "DLR", "O", "AVB",
        "EQR", "VTR", "SBAC", "WY", "ARE", "MAA", "UDR", "ESS", "INVH", "CPT",
        "VICI", "BXP", "KIM", "REG", "FRT", "HST", "IRM", "EXR", "CUBE", "LSI"
    ],
    "Communication Services": [
        "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR",
        "ATVI", "EA", "TTWO", "WBD", "PARA", "FOX", "FOXA", "NWSA", "NWS", "IPG",
        "OMC", "LYV", "MTCH", "ZG", "PINS", "SNAP", "RBLX", "ROKU", "SPOT", "TWTR"
    ]
}


# Build complete S&P 500 list from sectors
SP500 = []
for sector_stocks in SP500_BY_SECTOR.values():
    SP500.extend(sector_stocks)
SP500 = list(set(SP500))  # Remove duplicates


# High Volume / Most Active Stocks
MOST_ACTIVE = [
    "AAPL", "TSLA", "NVDA", "AMD", "AMZN", "META", "MSFT", "GOOGL", "NFLX", "SPY",
    "QQQ", "INTC", "BAC", "F", "PLTR", "SOFI", "NIO", "RIVN", "LCID", "PLUG",
    "COIN", "HOOD", "SQ", "PYPL", "SHOP", "RBLX", "SNAP", "UBER", "LYFT", "ABNB"
]


# Small Cap Growth Stocks
SMALL_CAP_GROWTH = [
    "UPST", "AFRM", "SOFI", "HOOD", "COIN", "RBLX", "DKNG", "PENN", "CRSR", "ASAN",
    "MDB", "NET", "DDOG", "ZS", "CRWD", "OKTA", "SNOW", "PATH", "U", "CFLT",
    "BILL", "HUBS", "TWLO", "DOCN", "GTLB", "MNDY", "FROG", "BRZE", "APP", "IS"
]


# Dividend Aristocrats (25+ years of dividend increases)
DIVIDEND_ARISTOCRATS = [
    "MMM", "ABT", "ABBV", "AFL", "APD", "ALB", "AMCR", "AOS", "ATO", "BDX",
    "BRO", "CAH", "CAT", "CVX", "CINF", "CTAS", "CLX", "KO", "CL", "ED",
    "DOV", "EMR", "ESS", "EXPD", "XOM", "FRT", "BEN", "GD", "GPC", "GWW",
    "HRL", "ITW", "JNJ", "KMB", "LEG", "LIN", "LOW", "MCD", "MDT", "MKC",
    "NEE", "NUE", "PNR", "PPG", "PG", "O", "ROP", "SPGI", "SHW", "SWK",
    "SYY", "TGT", "T", "TROW", "ADP", "WMT", "WBA", "WST", "ECL", "CB"
]


def get_sp500_symbols() -> list[str]:
    """Get S&P 500 component symbols."""
    return SP500.copy()


def get_nasdaq100_symbols() -> list[str]:
    """Get NASDAQ 100 component symbols."""
    return NASDAQ_100.copy()


def get_dow30_symbols() -> list[str]:
    """Get Dow Jones 30 component symbols."""
    return DOW_30.copy()


def get_sector_stocks(sector: str) -> list[str]:
    """Get stocks for a specific sector."""
    return SP500_BY_SECTOR.get(sector, []).copy()


def get_dividend_aristocrats() -> list[str]:
    """Get Dividend Aristocrat stocks."""
    return DIVIDEND_ARISTOCRATS.copy()


def get_small_cap_growth() -> list[str]:
    """Get small cap growth stocks."""
    return SMALL_CAP_GROWTH.copy()


def get_most_active() -> list[str]:
    """Get most actively traded stocks."""
    return MOST_ACTIVE.copy()


def get_all_sectors() -> list[str]:
    """Get list of all sectors."""
    return list(SP500_BY_SECTOR.keys())


def get_all_major_stocks() -> list[str]:
    """
    Get all major stocks from NYSE and NASDAQ.

    Combines S&P 500, NASDAQ 100, and other major stocks.
    Returns deduplicated list.
    """
    all_stocks = set()
    all_stocks.update(SP500)
    all_stocks.update(NASDAQ_100)
    all_stocks.update(DOW_30)
    all_stocks.update(DIVIDEND_ARISTOCRATS)
    return sorted(list(all_stocks))


def get_stocks_by_universe(universe: str) -> list[str]:
    """
    Get stocks by universe name.

    Args:
        universe: One of 'sp500', 'nasdaq100', 'dow30', 'dividend', 'small_cap', 'all'

    Returns:
        List of stock symbols
    """
    universes = {
        "sp500": get_sp500_symbols,
        "nasdaq100": get_nasdaq100_symbols,
        "dow30": get_dow30_symbols,
        "dividend": get_dividend_aristocrats,
        "small_cap": get_small_cap_growth,
        "most_active": get_most_active,
        "all": get_all_major_stocks,
    }

    func = universes.get(universe.lower())
    if func:
        return func()
    return []


def search_symbols(query: str, limit: int = 10) -> list[str]:
    """
    Search for symbols matching a query.

    Args:
        query: Search query (symbol prefix)
        limit: Maximum results to return

    Returns:
        List of matching symbols
    """
    query = query.upper()
    all_symbols = get_all_major_stocks()

    # Exact matches first
    exact = [s for s in all_symbols if s == query]

    # Prefix matches
    prefix = [s for s in all_symbols if s.startswith(query) and s != query]

    # Contains matches
    contains = [s for s in all_symbols if query in s and s not in exact and s not in prefix]

    results = exact + prefix + contains
    return results[:limit]
