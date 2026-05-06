from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import re
import unicodedata


Orientation = Literal["horizontal", "vertical"]
TriangleSide = Literal["left"]


@dataclass(frozen=True)
class StripeFlag:
    country: str
    orientation: Orientation
    colors: tuple[str, ...]
    triangle_color: str | None = None
    triangle_side: TriangleSide | None = None
    triangle_width_fraction: float = 0.5


@dataclass(frozen=True)
class ImageFlag:
    country: str
    code: str
    image_path: str
    source_url: str


FlagSpec = StripeFlag | ImageFlag


FLAG_IMAGE_ROOT = Path("assets") / "flags" / "world_rectangle"


COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "blue": (0, 85, 164),
    "brown": (122, 74, 36),
    "cyan": (0, 180, 216),
    "gold": (232, 183, 33),
    "gray": (128, 128, 128),
    "green": (0, 135, 81),
    "light_blue": (110, 187, 233),
    "lime": (129, 199, 62),
    "magenta": (208, 70, 153),
    "maroon": (124, 35, 74),
    "navy": (36, 61, 130),
    "orange": (255, 136, 62),
    "pink": (255, 143, 171),
    "purple": (111, 66, 193),
    "red": (206, 17, 38),
    "teal": (0, 133, 119),
    "violet": (143, 110, 201),
    "white": (255, 255, 255),
    "yellow": (255, 205, 0),
}

STRIPE_FLAGS: tuple[StripeFlag, ...] = (
    StripeFlag("Armenia", "horizontal", ("red", "blue", "orange")),
    StripeFlag("Austria", "horizontal", ("red", "white", "red")),
    StripeFlag("Belgium", "vertical", ("black", "yellow", "red")),
    StripeFlag("Bulgaria", "horizontal", ("white", "green", "red")),
    StripeFlag("Chad", "vertical", ("blue", "yellow", "red")),
    StripeFlag("Cote d'Ivoire", "vertical", ("orange", "white", "green")),
    StripeFlag("Estonia", "horizontal", ("blue", "black", "white")),
    StripeFlag("France", "vertical", ("blue", "white", "red")),
    StripeFlag("Gabon", "horizontal", ("green", "yellow", "blue")),
    StripeFlag("Germany", "horizontal", ("black", "red", "yellow")),
    StripeFlag("Guinea", "vertical", ("red", "yellow", "green")),
    StripeFlag("Hungary", "horizontal", ("red", "white", "green")),
    StripeFlag("Ireland", "vertical", ("green", "white", "orange")),
    StripeFlag("Italy", "vertical", ("green", "white", "red")),
    StripeFlag("Lithuania", "horizontal", ("yellow", "green", "red")),
    StripeFlag("Luxembourg", "horizontal", ("red", "white", "light_blue")),
    StripeFlag("Mali", "vertical", ("green", "yellow", "red")),
    StripeFlag("Netherlands", "horizontal", ("red", "white", "blue")),
    StripeFlag("Nigeria", "vertical", ("green", "white", "green")),
    StripeFlag("Peru", "vertical", ("red", "white", "red")),
    StripeFlag("Romania", "vertical", ("blue", "yellow", "red")),
    StripeFlag("Russia", "horizontal", ("white", "blue", "red")),
    StripeFlag("Sierra Leone", "horizontal", ("green", "white", "blue")),
    StripeFlag("Ukraine", "horizontal", ("blue", "yellow")),
    StripeFlag("Yemen", "horizontal", ("red", "white", "black")),
)

# Real hoist-triangle flags that fit the current renderer without needing extra
# stars, emblems, or non-triangular shapes.
REAL_TRIANGLE_FLAGS: tuple[StripeFlag, ...] = (
    StripeFlag(
        "Bahamas",
        "horizontal",
        ("light_blue", "yellow", "light_blue"),
        triangle_color="black",
        triangle_side="left",
    ),
    StripeFlag(
        "Czech Republic",
        "horizontal",
        ("white", "red"),
        triangle_color="blue",
        triangle_side="left",
    ),
    StripeFlag(
        "Palestine",
        "horizontal",
        ("black", "white", "green"),
        triangle_color="red",
        triangle_side="left",
    ),
    StripeFlag(
        "Sudan",
        "horizontal",
        ("red", "white", "black"),
        triangle_color="green",
        triangle_side="left",
    ),
)

BASE_FLAG_BY_COUNTRY: dict[str, StripeFlag] = {flag.country: flag for flag in STRIPE_FLAGS}
REAL_TRIANGLE_FLAG_BY_COUNTRY: dict[str, StripeFlag] = {
    flag.country: flag for flag in REAL_TRIANGLE_FLAGS
}
SUPPORTED_FLAG_BY_COUNTRY: dict[str, StripeFlag] = {
    **BASE_FLAG_BY_COUNTRY,
    **REAL_TRIANGLE_FLAG_BY_COUNTRY,
}

WORLD_RECTANGLE_COUNTRIES: tuple[str, ...] = (
    "Afghanistan",
    "Albania",
    "Algeria",
    "Andorra",
    "Angola",
    "Antigua and Barbuda",
    "Argentina",
    "Armenia",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahamas",
    "Bahrain",
    "Bangladesh",
    "Barbados",
    "Belarus",
    "Belgium",
    "Belize",
    "Benin",
    "Bhutan",
    "Bolivia",
    "Bosnia and Herzegovina",
    "Botswana",
    "Brazil",
    "Brunei",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    "Cabo Verde",
    "Cambodia",
    "Cameroon",
    "Canada",
    "Central African Republic",
    "Chad",
    "Chile",
    "China",
    "Colombia",
    "Comoros",
    "Congo",
    "Costa Rica",
    "Cote d'Ivoire",
    "Croatia",
    "Cuba",
    "Cyprus",
    "Czech Republic",
    "Democratic Republic of the Congo",
    "Denmark",
    "Djibouti",
    "Dominica",
    "Dominican Republic",
    "Ecuador",
    "Egypt",
    "El Salvador",
    "Equatorial Guinea",
    "Eritrea",
    "Estonia",
    "Eswatini",
    "Ethiopia",
    "Fiji",
    "Finland",
    "France",
    "Gabon",
    "Gambia",
    "Georgia",
    "Germany",
    "Ghana",
    "Greece",
    "Grenada",
    "Guatemala",
    "Guinea",
    "Guinea-Bissau",
    "Guyana",
    "Haiti",
    "Honduras",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Jamaica",
    "Japan",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kiribati",
    "Kuwait",
    "Kyrgyzstan",
    "Laos",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Liberia",
    "Libya",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Maldives",
    "Mali",
    "Malta",
    "Marshall Islands",
    "Mauritania",
    "Mauritius",
    "Mexico",
    "Micronesia",
    "Moldova",
    "Monaco",
    "Mongolia",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Myanmar",
    "Namibia",
    "Nauru",
    "Netherlands",
    "New Zealand",
    "Nicaragua",
    "Niger",
    "Nigeria",
    "North Korea",
    "North Macedonia",
    "Norway",
    "Oman",
    "Pakistan",
    "Palau",
    "Palestine",
    "Panama",
    "Papua New Guinea",
    "Paraguay",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "Russia",
    "Rwanda",
    "Saint Kitts and Nevis",
    "Saint Lucia",
    "Saint Vincent and the Grenadines",
    "Samoa",
    "San Marino",
    "Sao Tome and Principe",
    "Saudi Arabia",
    "Senegal",
    "Serbia",
    "Seychelles",
    "Sierra Leone",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "Solomon Islands",
    "Somalia",
    "South Africa",
    "South Korea",
    "South Sudan",
    "Spain",
    "Sri Lanka",
    "Sudan",
    "Suriname",
    "Sweden",
    "Switzerland",
    "Syria",
    "Tajikistan",
    "Tanzania",
    "Thailand",
    "Timor-Leste",
    "Togo",
    "Tonga",
    "Trinidad and Tobago",
    "Tunisia",
    "Turkey",
    "Turkmenistan",
    "Tuvalu",
    "Uganda",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Uruguay",
    "Uzbekistan",
    "Vanuatu",
    "Vatican City",
    "Venezuela",
    "Vietnam",
    "Yemen",
    "Zambia",
    "Zimbabwe",
)

COUNTRY_NAME_ALIASES: dict[str, str] = {
    "america": "United States",
    "bolivia plurinational state of": "Bolivia",
    "burma": "Myanmar",
    "cape verde": "Cabo Verde",
    "cote divoire": "Cote d'Ivoire",
    "czechia": "Czech Republic",
    "democratic peoples republic of korea": "North Korea",
    "dprk": "North Korea",
    "drc": "Democratic Republic of the Congo",
    "east timor": "Timor-Leste",
    "federated states of micronesia": "Micronesia",
    "great britain": "United Kingdom",
    "holy see": "Vatican City",
    "iran islamic republic of": "Iran",
    "ivory coast": "Cote d'Ivoire",
    "lao peoples democratic republic": "Laos",
    "moldova republic of": "Moldova",
    "palestinian territories": "Palestine",
    "republic of congo": "Congo",
    "republic of korea": "South Korea",
    "russian federation": "Russia",
    "south korea republic of korea": "South Korea",
    "state of palestine": "Palestine",
    "swaziland": "Eswatini",
    "syria arab republic": "Syria",
    "taiwan": "Taiwan",
    "the bahamas": "Bahamas",
    "the gambia": "Gambia",
    "turkiye": "Turkey",
    "u s": "United States",
    "u s a": "United States",
    "uk": "United Kingdom",
    "united states of america": "United States",
    "usa": "United States",
    "venezuela bolivarian republic of": "Venezuela",
    "viet nam": "Vietnam",
}

EXTRA_RECTANGLE_COUNTRY_LIKE_NAMES: tuple[str, ...] = (
    "Kosovo",
    "Taiwan",
    "Western Sahara",
)

WORLD_RECTANGLE_FLAG_CODES: dict[str, str] = {
    "Afghanistan": "af",
    "Albania": "al",
    "Algeria": "dz",
    "Andorra": "ad",
    "Angola": "ao",
    "Antigua and Barbuda": "ag",
    "Argentina": "ar",
    "Armenia": "am",
    "Australia": "au",
    "Austria": "at",
    "Azerbaijan": "az",
    "Bahamas": "bs",
    "Bahrain": "bh",
    "Bangladesh": "bd",
    "Barbados": "bb",
    "Belarus": "by",
    "Belgium": "be",
    "Belize": "bz",
    "Benin": "bj",
    "Bhutan": "bt",
    "Bolivia": "bo",
    "Bosnia and Herzegovina": "ba",
    "Botswana": "bw",
    "Brazil": "br",
    "Brunei": "bn",
    "Bulgaria": "bg",
    "Burkina Faso": "bf",
    "Burundi": "bi",
    "Cabo Verde": "cv",
    "Cambodia": "kh",
    "Cameroon": "cm",
    "Canada": "ca",
    "Central African Republic": "cf",
    "Chad": "td",
    "Chile": "cl",
    "China": "cn",
    "Colombia": "co",
    "Comoros": "km",
    "Congo": "cg",
    "Costa Rica": "cr",
    "Cote d'Ivoire": "ci",
    "Croatia": "hr",
    "Cuba": "cu",
    "Cyprus": "cy",
    "Czech Republic": "cz",
    "Democratic Republic of the Congo": "cd",
    "Denmark": "dk",
    "Djibouti": "dj",
    "Dominica": "dm",
    "Dominican Republic": "do",
    "Ecuador": "ec",
    "Egypt": "eg",
    "El Salvador": "sv",
    "Equatorial Guinea": "gq",
    "Eritrea": "er",
    "Estonia": "ee",
    "Eswatini": "sz",
    "Ethiopia": "et",
    "Fiji": "fj",
    "Finland": "fi",
    "France": "fr",
    "Gabon": "ga",
    "Gambia": "gm",
    "Georgia": "ge",
    "Germany": "de",
    "Ghana": "gh",
    "Greece": "gr",
    "Grenada": "gd",
    "Guatemala": "gt",
    "Guinea": "gn",
    "Guinea-Bissau": "gw",
    "Guyana": "gy",
    "Haiti": "ht",
    "Honduras": "hn",
    "Hungary": "hu",
    "Iceland": "is",
    "India": "in",
    "Indonesia": "id",
    "Iran": "ir",
    "Iraq": "iq",
    "Ireland": "ie",
    "Israel": "il",
    "Italy": "it",
    "Jamaica": "jm",
    "Japan": "jp",
    "Jordan": "jo",
    "Kazakhstan": "kz",
    "Kenya": "ke",
    "Kiribati": "ki",
    "Kuwait": "kw",
    "Kyrgyzstan": "kg",
    "Laos": "la",
    "Latvia": "lv",
    "Lebanon": "lb",
    "Lesotho": "ls",
    "Liberia": "lr",
    "Libya": "ly",
    "Liechtenstein": "li",
    "Lithuania": "lt",
    "Luxembourg": "lu",
    "Madagascar": "mg",
    "Malawi": "mw",
    "Malaysia": "my",
    "Maldives": "mv",
    "Mali": "ml",
    "Malta": "mt",
    "Marshall Islands": "mh",
    "Mauritania": "mr",
    "Mauritius": "mu",
    "Mexico": "mx",
    "Micronesia": "fm",
    "Moldova": "md",
    "Monaco": "mc",
    "Mongolia": "mn",
    "Montenegro": "me",
    "Morocco": "ma",
    "Mozambique": "mz",
    "Myanmar": "mm",
    "Namibia": "na",
    "Nauru": "nr",
    "Netherlands": "nl",
    "New Zealand": "nz",
    "Nicaragua": "ni",
    "Niger": "ne",
    "Nigeria": "ng",
    "North Korea": "kp",
    "North Macedonia": "mk",
    "Norway": "no",
    "Oman": "om",
    "Pakistan": "pk",
    "Palau": "pw",
    "Palestine": "ps",
    "Panama": "pa",
    "Papua New Guinea": "pg",
    "Paraguay": "py",
    "Peru": "pe",
    "Philippines": "ph",
    "Poland": "pl",
    "Portugal": "pt",
    "Qatar": "qa",
    "Romania": "ro",
    "Russia": "ru",
    "Rwanda": "rw",
    "Saint Kitts and Nevis": "kn",
    "Saint Lucia": "lc",
    "Saint Vincent and the Grenadines": "vc",
    "Samoa": "ws",
    "San Marino": "sm",
    "Sao Tome and Principe": "st",
    "Saudi Arabia": "sa",
    "Senegal": "sn",
    "Serbia": "rs",
    "Seychelles": "sc",
    "Sierra Leone": "sl",
    "Singapore": "sg",
    "Slovakia": "sk",
    "Slovenia": "si",
    "Solomon Islands": "sb",
    "Somalia": "so",
    "South Africa": "za",
    "South Korea": "kr",
    "South Sudan": "ss",
    "Spain": "es",
    "Sri Lanka": "lk",
    "Sudan": "sd",
    "Suriname": "sr",
    "Sweden": "se",
    "Switzerland": "ch",
    "Syria": "sy",
    "Tajikistan": "tj",
    "Tanzania": "tz",
    "Thailand": "th",
    "Timor-Leste": "tl",
    "Togo": "tg",
    "Tonga": "to",
    "Trinidad and Tobago": "tt",
    "Tunisia": "tn",
    "Turkey": "tr",
    "Turkmenistan": "tm",
    "Tuvalu": "tv",
    "Uganda": "ug",
    "Ukraine": "ua",
    "United Arab Emirates": "ae",
    "United Kingdom": "gb",
    "United States": "us",
    "Uruguay": "uy",
    "Uzbekistan": "uz",
    "Vanuatu": "vu",
    "Vatican City": "va",
    "Venezuela": "ve",
    "Vietnam": "vn",
    "Yemen": "ye",
    "Zambia": "zm",
    "Zimbabwe": "zw",
    "Kosovo": "xk",
    "Taiwan": "tw",
    "Western Sahara": "eh",
}

COUNTRY_NAME_UNIVERSES: dict[str, tuple[str, ...]] = {
    "world_rectangle": WORLD_RECTANGLE_COUNTRIES + EXTRA_RECTANGLE_COUNTRY_LIKE_NAMES,
}

COUNTRY_POOLS: dict[str, tuple[str, ...]] = {
    "stripe_easy_14": (
        "Austria",
        "Belgium",
        "Chad",
        "France",
        "Germany",
        "Guinea",
        "Ireland",
        "Italy",
        "Luxembourg",
        "Mali",
        "Netherlands",
        "Nigeria",
        "Russia",
        "Ukraine",
    ),
    "stripe_easy_15_legacy": (
        "Austria",
        "Belgium",
        "Chad",
        "France",
        "Germany",
        "Guinea",
        "Ireland",
        "Italy",
        "Luxembourg",
        "Mali",
        "Netherlands",
        "Nigeria",
        "Romania",
        "Russia",
        "Ukraine",
    ),
    "stripe_expanded_24": tuple(flag.country for flag in STRIPE_FLAGS if flag.country != "Romania"),
    "stripe_expanded_25_legacy": tuple(flag.country for flag in STRIPE_FLAGS),
    # All currently rendered flag countries whose designs are made only from
    # rectangular color regions. This is a renderable truth pool, not the full
    # world list of rectangular national flags.
    "rendered_rectangle_25": tuple(flag.country for flag in STRIPE_FLAGS),
    "world_rectangle_images": COUNTRY_NAME_UNIVERSES["world_rectangle"],
    "real_triangle_4": tuple(flag.country for flag in REAL_TRIANGLE_FLAGS),
    "stripe_plus_real_triangle_28": tuple(flag.country for flag in STRIPE_FLAGS if flag.country != "Romania")
    + tuple(flag.country for flag in REAL_TRIANGLE_FLAGS),
}


def _normalize_country_name(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    ascii_value = ascii_value.replace("&", " and ")
    ascii_value = re.sub(r"[^A-Za-z0-9]+", " ", ascii_value)
    ascii_value = re.sub(r"\s+", " ", ascii_value).strip().lower()
    if ascii_value.startswith("the "):
        ascii_value = ascii_value[4:]
    return ascii_value


def canonical_country_name(
    value: str,
    *,
    universe_name: str = "world_rectangle",
) -> str | None:
    if universe_name not in COUNTRY_NAME_UNIVERSES:
        valid = ", ".join(sorted(COUNTRY_NAME_UNIVERSES))
        raise KeyError(f"Unknown country name universe {universe_name!r}; choose from: {valid}")
    normalized = _normalize_country_name(value)
    if not normalized:
        return None
    if normalized in COUNTRY_NAME_ALIASES:
        alias_target = COUNTRY_NAME_ALIASES[normalized]
        if alias_target in COUNTRY_NAME_UNIVERSES[universe_name]:
            return alias_target
        return None
    by_normalized = {
        _normalize_country_name(country): country
        for country in COUNTRY_NAME_UNIVERSES[universe_name]
    }
    if normalized in by_normalized:
        return by_normalized[normalized]
    return None


def image_flag_source_url(code: str, *, width: int = 640) -> str:
    return f"https://flagcdn.com/w{width}/{code.lower()}.png"


def image_flag_asset_path(code: str, *, root: Path | None = None) -> Path:
    base = FLAG_IMAGE_ROOT if root is None else root
    return base / f"{code.lower()}.png"


def image_flag_for_country(country: str, *, root: Path | None = None) -> ImageFlag:
    try:
        code = WORLD_RECTANGLE_FLAG_CODES[country]
    except KeyError as exc:
        raise KeyError(f"No image-backed flag code for country: {country}") from exc
    return ImageFlag(
        country=country,
        code=code,
        image_path=str(image_flag_asset_path(code, root=root)),
        source_url=image_flag_source_url(code),
    )


def get_flag(country: str) -> FlagSpec:
    try:
        return SUPPORTED_FLAG_BY_COUNTRY[country]
    except KeyError:
        return image_flag_for_country(country)


def get_country_lookup(pool_name: str) -> dict[str, FlagSpec]:
    if pool_name not in COUNTRY_POOLS:
        valid = ", ".join(sorted(COUNTRY_POOLS))
        raise KeyError(f"Unknown country pool {pool_name!r}; choose from: {valid}")
    if pool_name == "world_rectangle_images":
        return {
            country: image_flag_for_country(country)
            for country in COUNTRY_POOLS[pool_name]
        }
    if pool_name in {"real_triangle_4", "stripe_plus_real_triangle_28"}:
        lookup = SUPPORTED_FLAG_BY_COUNTRY
    else:
        lookup = BASE_FLAG_BY_COUNTRY
    return {country: lookup[country] for country in COUNTRY_POOLS[pool_name]}


def get_country_pool(pool_name: str) -> list[FlagSpec]:
    return list(get_country_lookup(pool_name).values())
