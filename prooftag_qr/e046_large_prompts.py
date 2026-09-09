"""Deterministic prompt source for the E046 large advisor catalogue.

The 256 final prompt strings are deliberately assembled from 256 hand-curated
subjects and the 16-row structural design declared in :mod:`e046_large_catalog`.
There is no model call, clock, process randomness, or environment dependency in
this module.  The resulting JSON catalogue is therefore reproducible byte for
byte while every family still contains genuinely different scenes.
"""

from __future__ import annotations

from typing import Final

from .e046_large_catalog import ANCHOR_PROMPT_TEXT, TAG_DESIGN

Scene = tuple[str, str, str]


# Each tuple is (globally unique slug, subject clause, rendering medium).  The
# historical brutalist anchor sits on design row 5 (medium frequency, symmetric,
# mixed light, low contrast, muted, generous negative space, fine, centered): the
# only row whose tags match its verbatim historical wording.  Colour,
# light and composition are intentionally supplied by TAG_DESIGN rather than
# baked into the subject, so those experimental factors do not contradict the
# scene wording.
FAMILY_SCENES: Final[dict[str, tuple[Scene, ...]]] = {
    "architectural_grid": (
        ("canal_row_houses", "narrow canal houses reflected in still water", "urban photograph"),
        ("cliffside_monastery", "a cliffside monastery of stacked arcades", "travel photograph"),
        (
            "museum_coffered_atrium",
            "a museum atrium beneath a coffered roof",
            "interior photograph",
        ),
        (
            "desert_observatory",
            "a stepped desert observatory and its deep portals",
            "architectural photograph",
        ),
        (
            "brutalist_courtyard",
            "a monumental brutalist courtyard with repeated square windows",
            "architectural photograph",
        ),
        (
            "concrete_library",
            "a concrete library with tiered reading galleries",
            "interior photograph",
        ),
        (
            "alpine_train_station",
            "an alpine train station beneath timber trusses",
            "documentary photograph",
        ),
        (
            "terraced_thermal_baths",
            "terraced thermal baths cut into stone",
            "architectural photograph",
        ),
        (
            "modernist_cloister",
            "a modernist cloister surrounding a shallow pool",
            "editorial photograph",
        ),
        ("bamboo_pavilion", "an open bamboo pavilion with modular bays", "design photograph"),
        (
            "circular_archive",
            "a circular archive lined with recessed alcoves",
            "interior photograph",
        ),
        (
            "volcanic_chapel",
            "a small chapel built from volcanic blocks",
            "architectural photograph",
        ),
        ("elevated_housing", "elevated housing blocks connected by walkways", "urban photograph"),
        (
            "underground_cistern",
            "an underground cistern filled with stone columns",
            "heritage photograph",
        ),
        (
            "wooden_market_hall",
            "a wooden market hall under folded beams",
            "architectural photograph",
        ),
        ("coastal_fortress", "a coastal fortress of ramps and courtyards", "aerial photograph"),
    ),
    "industrial_mechanical": (
        (
            "turbine_hall",
            "a hydroelectric turbine hall with aligned generators",
            "industrial photograph",
        ),
        (
            "watch_escapement",
            "a mechanical watch escapement seen at close range",
            "macro photograph",
        ),
        (
            "copper_pipe_plant",
            "an old processing plant threaded with copper pipes",
            "industrial photograph",
        ),
        (
            "robotic_assembly",
            "robotic arms working around an assembly fixture",
            "documentary photograph",
        ),
        (
            "locomotive_valves",
            "steam locomotive valves and connecting rods",
            "mechanical photograph",
        ),
        (
            "satellite_workbench",
            "a satellite chassis on an engineering workbench",
            "technical photograph",
        ),
        ("weaving_loom", "a mechanical weaving loom with moving heddles", "workshop photograph"),
        (
            "ship_engine_room",
            "a ship engine room crossed by service ladders",
            "industrial photograph",
        ),
        (
            "analog_synth_rack",
            "an analog synthesizer rack of knobs and patch cables",
            "studio photograph",
        ),
        (
            "printing_press",
            "a letterpress mechanism with rollers and flywheels",
            "workshop photograph",
        ),
        (
            "solar_tracking_array",
            "solar trackers and their articulated joints",
            "engineering photograph",
        ),
        (
            "aircraft_hydraulics",
            "aircraft hydraulic lines inside an open service bay",
            "technical photograph",
        ),
        ("clocktower_gears", "clocktower gears behind a cast iron frame", "heritage photograph"),
        (
            "server_cooling",
            "liquid cooling manifolds inside a computing rack",
            "technical photograph",
        ),
        (
            "ceramic_kiln_controls",
            "ceramic kiln controls beside insulated ducts",
            "workshop photograph",
        ),
        (
            "harbor_cranes",
            "harbor cranes overlapping above loading machinery",
            "documentary photograph",
        ),
    ),
    "geometric_ornamental": (
        (
            "zellige_fountain",
            "a courtyard fountain surrounded by zellige mosaic",
            "design photograph",
        ),
        ("rose_window", "a carved stone rose window with radial tracery", "heritage photograph"),
        ("parquet_medallion", "an intricate parquet floor medallion", "interior photograph"),
        ("lacquer_screen", "a folding lacquer screen with geometric inlay", "museum photograph"),
        (
            "stained_glass_vault",
            "a vaulted canopy of stained glass panels",
            "architectural photograph",
        ),
        (
            "islamic_lattice",
            "a pierced stone lattice casting patterned shadows",
            "heritage photograph",
        ),
        ("paper_rosette", "a layered paper rosette built from folded facets", "craft photograph"),
        (
            "ceramic_tile_stair",
            "a staircase faced with mismatched ceramic tiles",
            "editorial photograph",
        ),
        ("metal_ceiling", "an embossed metal ceiling of repeating coffers", "interior photograph"),
        ("garden_paving", "garden paving arranged as interlocking polygons", "aerial photograph"),
        ("beaded_collar", "a ceremonial collar assembled from tiny beads", "museum photograph"),
        ("wood_marquetry", "wood marquetry forming nested star polygons", "object photograph"),
        ("festival_lanterns", "folded festival lanterns suspended in a canopy", "night photograph"),
        ("granite_labyrinth", "a granite labyrinth set into a public plaza", "aerial photograph"),
        (
            "plaster_muqarnas",
            "a plaster niche filled with faceted muqarnas",
            "architectural photograph",
        ),
        ("shell_inlay_box", "a small box covered in shell tessellation", "product photograph"),
    ),
    "textile_pattern": (
        ("indigo_loom", "indigo threads stretched across a hand loom", "textile photograph"),
        ("wool_quilt", "a patchwork quilt assembled from wool panels", "craft photograph"),
        ("basket_weave", "split reeds crossing through a basket weave", "macro photograph"),
        ("silk_pleats", "pleated silk forming a sculptural fan", "fashion photograph"),
        ("knitted_cable", "a thick knitted cable twisting through soft yarn", "macro photograph"),
        (
            "embroidered_meadow",
            "an embroidered meadow of layered botanical stitches",
            "craft photograph",
        ),
        ("woven_wall_hanging", "a woven wall hanging with raised fibres", "interior photograph"),
        ("lace_collar", "a handmade lace collar spread across linen", "object photograph"),
        ("sashiko_panel", "a sashiko panel joined by running stitches", "textile photograph"),
        ("velvet_drapery", "heavy velvet drapery falling into deep folds", "fashion photograph"),
        ("macrame_arch", "a knotted macrame arch with long tassels", "craft photograph"),
        ("felt_topography", "cut felt layers resembling a topographic relief", "design photograph"),
        ("ikat_ribbons", "ikat ribbons hanging in overlapping bands", "textile photograph"),
        ("crochet_coral", "crocheted forms resembling a coral colony", "studio photograph"),
        ("linen_samples", "linen samples arranged across a work table", "editorial photograph"),
        ("braided_rope", "braided marine rope coiled into broad loops", "object photograph"),
    ),
    "botanical_rows": (
        (
            "vineyard_terraces",
            "vineyard terraces descending along a hillside",
            "landscape photograph",
        ),
        (
            "greenhouse_benches",
            "greenhouse benches filled with young seedlings",
            "botanical photograph",
        ),
        ("lavender_fields", "long lavender rows following rolling ground", "aerial photograph"),
        (
            "citrus_orchard",
            "a citrus orchard crossed by irrigation channels",
            "documentary photograph",
        ),
        ("rice_paddies", "rice paddies stepping around a rural valley", "aerial photograph"),
        (
            "succulent_nursery",
            "a succulent nursery arranged on narrow shelves",
            "botanical photograph",
        ),
        ("tea_plantation", "tea bushes clipped into sweeping contours", "landscape photograph"),
        ("tulip_beds", "formal tulip beds divided by gravel paths", "garden photograph"),
        ("olive_grove", "an old olive grove planted on dry terraces", "documentary photograph"),
        (
            "herbarium_drawers",
            "open herbarium drawers holding pressed plants",
            "archive photograph",
        ),
        ("apple_espalier", "apple trees trained along parallel wires", "horticultural photograph"),
        (
            "floating_gardens",
            "floating vegetable gardens separated by waterways",
            "aerial photograph",
        ),
        ("bonsai_benches", "bonsai specimens displayed on staggered benches", "garden photograph"),
        ("vertical_farm", "leafy crops growing inside a vertical farm", "industrial photograph"),
        (
            "cactus_conservatory",
            "columnar cacti rising through a conservatory",
            "botanical photograph",
        ),
        (
            "market_flower_crates",
            "flower crates ordered across a market floor",
            "documentary photograph",
        ),
    ),
    "organic_branching": (
        ("winter_oak", "the spreading crown of an old winter oak", "landscape photograph"),
        ("mangrove_roots", "mangrove roots dividing above tidal mud", "nature photograph"),
        ("river_delta", "a river delta splitting across coastal sand", "satellite photograph"),
        ("mycelium_network", "a mycelium network growing through forest soil", "macro photograph"),
        ("lightning_storm", "forked lightning crossing a distant storm", "night photograph"),
        ("leaf_veins", "translucent leaf veins seen from beneath", "macro photograph"),
        ("coral_tree", "a weathered coral skeleton with branching limbs", "museum photograph"),
        ("bronchial_cast", "a delicate bronchial cast suspended in space", "scientific photograph"),
        ("frost_dendrites", "frost dendrites spreading across a window", "macro photograph"),
        ("willow_canopy", "drooping willow branches surrounding a pond", "garden photograph"),
        ("neural_culture", "neurons extending across a laboratory culture", "microscopy image"),
        ("erosion_gullies", "erosion gullies branching down a clay slope", "aerial photograph"),
        ("root_bridge", "living tree roots woven into a footbridge", "documentary photograph"),
        ("fern_frond", "a fern frond unfurling into divided leaflets", "botanical photograph"),
        ("blood_vessels", "capillary vessels branching through living tissue", "scientific image"),
        ("driftwood_tree", "a bleached driftwood tree beside shallow water", "coastal photograph"),
    ),
    "underwater_organic": (
        (
            "kelp_cathedral",
            "towering kelp fronds forming an underwater cathedral",
            "marine photograph",
        ),
        ("sea_fan_garden", "sea fans growing across a reef wall", "underwater photograph"),
        ("jellyfish_bloom", "a bloom of jellyfish drifting in open water", "marine photograph"),
        ("anemone_pool", "sea anemones clustered inside a tide pool", "nature photograph"),
        ("seagrass_meadow", "seagrass bending across a shallow lagoon", "underwater photograph"),
        ("octopus_den", "an octopus emerging between volcanic rocks", "wildlife photograph"),
        ("sponge_towers", "tube sponges rising from a reef shelf", "marine photograph"),
        ("manta_procession", "manta rays gliding above a sandy channel", "underwater photograph"),
        ("coral_spawning", "coral colonies releasing clouds of spawn", "documentary photograph"),
        ("ice_divers", "divers moving beneath fractured sea ice", "expedition photograph"),
        ("nautilus_shells", "nautilus shells resting across the seabed", "still-life photograph"),
        ("hydrothermal_vents", "hydrothermal vents rising from the ocean floor", "deep-sea image"),
        ("schooling_sardines", "sardines turning as one dense school", "wildlife photograph"),
        (
            "mangrove_nursery",
            "young fish weaving among submerged mangrove roots",
            "marine photograph",
        ),
        (
            "moon_jelly_tank",
            "moon jellies floating inside a public aquarium",
            "aquarium photograph",
        ),
        ("reef_cave", "a reef cave opening toward sunlit water", "underwater photograph"),
    ),
    "crystal_mineral": (
        ("quartz_cluster", "a quartz cluster rising from raw stone", "mineral photograph"),
        ("agate_slice", "a polished agate slice revealing concentric bands", "macro photograph"),
        ("basalt_columns", "basalt columns descending toward the sea", "landscape photograph"),
        ("amethyst_geode", "an open amethyst geode lined with crystals", "studio photograph"),
        ("salt_flats", "salt polygons covering a dry lake bed", "aerial photograph"),
        ("ice_cave", "layered ice inside a glacial cave", "expedition photograph"),
        ("pyrite_cubes", "pyrite cubes embedded in dark matrix", "mineral photograph"),
        ("malachite_fold", "a malachite specimen with folded mineral bands", "macro photograph"),
        ("desert_rose", "desert rose crystals scattered over sand", "still-life photograph"),
        ("obsidian_shards", "obsidian shards arranged on volcanic ash", "object photograph"),
        ("calcite_cavern", "calcite formations inside a limestone cavern", "cave photograph"),
        ("mica_sheets", "thin mica sheets peeling from a specimen", "macro photograph"),
        ("bismuth_steps", "a bismuth crystal forming stepped hollows", "studio photograph"),
        ("river_pebbles", "wet river pebbles packed along a shore", "nature photograph"),
        (
            "selenite_needles",
            "selenite needles crossing inside a crystal pocket",
            "mineral photograph",
        ),
        ("meteorite_surface", "the pitted surface of an iron meteorite", "macro photograph"),
    ),
    "minimal_product": (
        ("cobalt_vase", "a single ceramic vase on a low plinth", "product photograph"),
        ("folded_headphones", "folded headphones resting on a plain table", "product photograph"),
        (
            "stone_perfume_bottle",
            "a sculptural perfume bottle beside one pebble",
            "studio photograph",
        ),
        ("wooden_camera", "a handmade wooden camera on seamless paper", "product photograph"),
        ("porcelain_teapot", "a porcelain teapot with a curved handle", "still-life photograph"),
        ("running_shoe", "one running shoe balanced above a pedestal", "advertising photograph"),
        ("desk_lamp", "a compact desk lamp casting one pool of light", "product photograph"),
        ("wristwatch_stone", "a wristwatch draped over a smooth stone", "studio photograph"),
        ("wireless_speaker", "a small wireless speaker beside folded fabric", "product photograph"),
        ("fountain_pen", "a fountain pen crossing an empty notebook", "still-life photograph"),
        ("ceramic_bowl", "a shallow ceramic bowl holding three pears", "editorial photograph"),
        ("hiking_thermos", "a metal thermos standing on a rock", "outdoor product photograph"),
        ("eyeglass_frames", "eyeglass frames folded beside a glass sphere", "product photograph"),
        ("leather_wallet", "an open leather wallet on a wooden block", "studio photograph"),
        ("handmade_scissors", "handmade scissors beside a spool of thread", "craft photograph"),
        ("portable_radio", "a portable radio isolated on a curved shelf", "product photograph"),
    ),
    "glass_translucent": (
        ("prism_still_life", "a glass prism splitting light across a table", "studio photograph"),
        ("blown_glass_vessels", "blown glass vessels grouped on a shelf", "object photograph"),
        ("soap_bubbles", "soap bubbles overlapping above a shallow dish", "macro photograph"),
        ("frosted_blocks", "frosted glass blocks forming a partition", "interior photograph"),
        ("rain_window", "rain droplets sliding down a window", "close-up photograph"),
        ("ice_sculpture", "a carved ice sculpture beginning to melt", "studio photograph"),
        (
            "laboratory_flasks",
            "laboratory flasks filled to different levels",
            "still-life photograph",
        ),
        (
            "glass_staircase",
            "a glass staircase suspended inside an atrium",
            "architectural photograph",
        ),
        ("resin_petals", "pressed petals trapped inside clear resin", "macro photograph"),
        (
            "crystal_chandelier",
            "a crystal chandelier seen directly from below",
            "interior photograph",
        ),
        (
            "water_caustics",
            "water caustics moving across a transparent vessel",
            "experimental photograph",
        ),
        (
            "glass_marble_field",
            "glass marbles scattered over a reflective surface",
            "product photograph",
        ),
        (
            "greenhouse_condensation",
            "condensation covering greenhouse panes",
            "botanical photograph",
        ),
        ("acrylic_ribbons", "transparent acrylic ribbons curling in space", "studio photograph"),
        (
            "museum_display_cases",
            "empty glass display cases aligned in a gallery",
            "interior photograph",
        ),
        ("frozen_air_bubbles", "air bubbles suspended inside lake ice", "nature photograph"),
    ),
    "celestial_space": (
        ("spiral_galaxy", "a spiral galaxy surrounded by distant stars", "astronomical image"),
        ("lunar_crater", "a lunar crater crossing the terminator", "telescope image"),
        ("aurora_valley", "an aurora arching above a mountain valley", "night photograph"),
        ("planetary_rings", "a ringed planet floating above its moon", "space illustration"),
        ("solar_prominence", "a solar prominence rising from the sun", "scientific image"),
        ("comet_tail", "a comet and its divided tail crossing space", "astronomical image"),
        ("nebula_pillars", "dust pillars inside a stellar nursery", "telescope image"),
        ("eclipse_corona", "the solar corona visible during an eclipse", "astronomy photograph"),
        ("meteor_shower", "a meteor shower above a quiet plateau", "night photograph"),
        ("cloud_planet", "a cloud-covered planet seen from orbit", "space illustration"),
        ("radio_dish_sky", "a radio telescope dish beneath the milky way", "landscape photograph"),
        ("binary_stars", "binary stars surrounded by luminous gas", "scientific illustration"),
        ("moon_phases", "successive moon phases arranged across the sky", "astronomy illustration"),
        ("cosmic_web", "filaments of galaxies spanning deep space", "scientific visualization"),
        ("rocket_plume", "a rocket plume expanding above the atmosphere", "space photograph"),
        ("stormy_jupiter", "swirling storms across a giant planet", "planetary image"),
    ),
    "landscape_negative_space": (
        ("foggy_headland", "a foggy headland fading into open sea", "landscape photograph"),
        ("single_desert_dune", "one desert dune beneath an empty sky", "fine-art photograph"),
        ("snow_fence", "a snow fence crossing a frozen plain", "winter photograph"),
        ("salt_lake_horizon", "a salt lake meeting a distant horizon", "landscape photograph"),
        ("misty_pine_island", "a small pine island emerging through mist", "nature photograph"),
        ("tidal_sandbar", "a tidal sandbar curving through shallow water", "aerial photograph"),
        ("prairie_silo", "a solitary grain silo on open prairie", "documentary photograph"),
        ("volcanic_ridge", "a volcanic ridge beneath drifting cloud", "landscape photograph"),
        ("reed_marsh", "sparse reeds standing across a quiet marsh", "nature photograph"),
        ("moonlit_iceberg", "one iceberg floating on a moonlit ocean", "night photograph"),
        ("empty_canyon_road", "an empty road winding through a canyon", "travel photograph"),
        ("grassland_boulder", "a lone boulder resting in tall grass", "fine-art photograph"),
        (
            "distant_waterfall",
            "a distant waterfall dropping through broad cliffs",
            "landscape photograph",
        ),
        ("cloud_shadow_fields", "cloud shadows drifting over open fields", "aerial photograph"),
        ("calm_lagoon", "a calm lagoon divided by a narrow spit", "coastal photograph"),
        ("desert_arch", "a natural stone arch surrounded by desert", "landscape photograph"),
    ),
    "macro_material": (
        ("oxidized_copper", "oxidized copper covered in layered patina", "macro photograph"),
        ("charred_wood", "charred wood splitting into angular plates", "macro photograph"),
        ("citrus_pulp", "translucent citrus pulp at extreme close range", "food macro photograph"),
        ("mossy_bark", "moss and lichen spreading over bark", "nature macro photograph"),
        ("cracked_clay", "sun-baked clay broken into irregular islands", "macro photograph"),
        ("foamed_metal", "porous foamed metal revealing connected cavities", "material photograph"),
        ("mother_of_pearl", "mother of pearl flashing across a shell", "macro photograph"),
        ("recycled_paper", "recycled paper fibres pressed into a sheet", "material photograph"),
        ("molten_wax", "molten wax cooling into overlapping ridges", "macro photograph"),
        (
            "granite_crystals",
            "granite crystals interlocking across a cut slab",
            "material photograph",
        ),
        ("bread_crumb", "an artisan bread crumb filled with open bubbles", "food photograph"),
        (
            "bird_feathers",
            "overlapping bird feathers seen at close range",
            "wildlife macro photograph",
        ),
        (
            "corroded_steel",
            "corroded steel peeling around old rivets",
            "industrial macro photograph",
        ),
        ("sea_foam", "sea foam forming cells across wet sand", "nature photograph"),
        ("folded_leather", "worn leather folding around deep creases", "material photograph"),
        (
            "silicon_wafer",
            "a silicon wafer revealing microscopic circuitry",
            "technical photograph",
        ),
    ),
    "urban_night": (
        ("rainy_tram_stop", "a rainy tram stop reflected in pavement", "street photograph"),
        ("elevated_highway", "an elevated highway crossing a sleeping city", "night photograph"),
        ("subway_platform", "an empty subway platform after midnight", "urban photograph"),
        (
            "rooftop_water_tanks",
            "rooftop water tanks against a luminous skyline",
            "city photograph",
        ),
        ("bicycle_alley", "bicycles parked along a narrow evening alley", "street photograph"),
        (
            "river_light_trails",
            "light trails flowing beside an urban river",
            "long-exposure photograph",
        ),
        (
            "glass_office_tower",
            "a glass office tower with scattered lit rooms",
            "architectural photograph",
        ),
        (
            "night_market_canopy",
            "a night market beneath overlapping canopies",
            "documentary photograph",
        ),
        ("parking_structure", "a parking structure illuminated from within", "urban photograph"),
        (
            "harbor_at_night",
            "working boats gathered inside a night harbor",
            "documentary photograph",
        ),
        ("pedestrian_bridge", "a pedestrian bridge spanning railway tracks", "night photograph"),
        (
            "apartment_balconies",
            "apartment balconies glowing at different depths",
            "city photograph",
        ),
        ("wet_crosswalk", "a wet crosswalk beneath changing traffic lights", "street photograph"),
        ("industrial_canal", "an industrial canal bordered by warehouses", "night photograph"),
        ("festival_square", "a public square filled with suspended lanterns", "urban photograph"),
        ("mountain_city", "a mountain city climbing through evening haze", "telephoto photograph"),
    ),
    "abstract_flow": (
        (
            "suminagashi_rings",
            "floating ink rings spreading across water",
            "experimental photograph",
        ),
        ("acrylic_pour", "liquid pigment folding through an acrylic pour", "abstract artwork"),
        ("smoke_ribbons", "smoke ribbons twisting through still air", "studio photograph"),
        (
            "magnetic_fluid",
            "magnetic fluid rising into clustered spikes",
            "experimental photograph",
        ),
        ("paper_topography", "cut paper contours forming an imaginary terrain", "paper artwork"),
        ("oil_interference", "oil interference patterns drifting across water", "macro photograph"),
        ("sand_vibration", "sand gathering along vibration nodes", "scientific photograph"),
        (
            "glass_color_fields",
            "transparent planes overlapping into color fields",
            "abstract photograph",
        ),
        ("marbled_stone_flow", "marbled stone veins sweeping across a slab", "material photograph"),
        (
            "generative_curves",
            "nested curves flowing around invisible attractors",
            "generative artwork",
        ),
        ("folded_mesh", "a flexible mesh folding into waves", "digital artwork"),
        ("ink_diffusion", "ink clouds diffusing through clear liquid", "experimental photograph"),
        ("raked_sand", "raked sand currents circling smooth stones", "garden photograph"),
        ("molten_glass_flow", "molten glass stretching into fluid strands", "workshop photograph"),
        ("moire_fabric", "layered mesh producing shifting moire fields", "abstract photograph"),
        (
            "wind_streamlines",
            "wind streamlines curling around geometric obstacles",
            "scientific visualization",
        ),
    ),
    "editorial_illustration": (
        (
            "paper_city_garden",
            "a cut-paper city transformed into a garden",
            "editorial illustration",
        ),
        ("library_whale", "a whale floating through a quiet library", "gouache illustration"),
        (
            "mechanical_orchard",
            "an orchard tended by gentle mechanical creatures",
            "ink illustration",
        ),
        (
            "mountain_teacup",
            "a mountain landscape contained inside a teacup",
            "collage illustration",
        ),
        ("night_bus_dream", "a night bus carrying a miniature dream world", "digital illustration"),
        ("seed_archive", "an underground archive protecting giant seeds", "editorial illustration"),
        ("cloud_factory", "a small factory producing drifting clouds", "screenprint illustration"),
        (
            "ocean_greenhouse",
            "an underwater greenhouse cultivated by divers",
            "gouache illustration",
        ),
        (
            "folding_observatory",
            "a portable observatory unfolding from a suitcase",
            "technical illustration",
        ),
        ("city_river_roots", "tree roots becoming rivers beneath a city", "linocut illustration"),
        (
            "ceramic_moon_market",
            "a moon market built from ceramic stalls",
            "editorial illustration",
        ),
        (
            "migrating_houses",
            "small houses migrating across an open plain",
            "watercolor illustration",
        ),
        ("clockwork_wetland", "a wetland shared by birds and clockwork reeds", "ink illustration"),
        ("solar_kite_village", "a village powered by enormous solar kites", "digital illustration"),
        ("museum_of_rain", "a museum collecting different forms of rain", "collage illustration"),
        (
            "forest_telescope",
            "a giant telescope growing through a forest",
            "editorial illustration",
        ),
    ),
}


SPATIAL = {
    "low": "broad simple forms",
    "medium": "balanced mid-scale detail",
    "high": "intricate fine detail",
}
SYMMETRY = {
    "symmetric": "clear bilateral symmetry",
    "asymmetric": "deliberately asymmetric balance",
}
BRIGHTNESS = {
    "light": "bright airy illumination",
    "dark": "deep low-key illumination",
    "mixed": "mixed light and shadow",
}
CONTRAST = {"low": "gentle tonal contrast", "high": "strong tonal contrast"}
COLORFULNESS = {
    "muted": "restrained muted palette",
    "colorful": "rich varied colors",
    "monochrome": "strict monochrome palette",
}
NEGATIVE_SPACE = {
    "low": "densely filled frame",
    "medium": "balanced open space",
    "high": "generous calm negative space",
}
TEXTURE = {"fine": "fine-grained texture", "coarse": "broad coarse texture"}
COMPOSITION = {
    "centered": "centered composition",
    "off_center": "off-center composition",
    "repetitive": "rhythmic repeated composition",
}


def _prompt(subject: str, medium: str, row: int) -> str:
    tags = TAG_DESIGN[row - 1]
    return ", ".join(
        (
            subject,
            medium,
            SPATIAL[tags["spatial_frequency"]],
            SYMMETRY[tags["symmetry"]],
            BRIGHTNESS[tags["brightness"]],
            CONTRAST[tags["contrast"]],
            COLORFULNESS[tags["colorfulness"]],
            NEGATIVE_SPACE[tags["negative_space"]],
            TEXTURE[tags["texture_density"]],
            COMPOSITION[tags["composition_type"]],
            "no lettering",
        )
    )


def _build_family(scenes: tuple[Scene, ...]) -> tuple[dict[str, object], ...]:
    if len(scenes) != 16:
        raise ValueError(f"each family must define 16 scenes, got {len(scenes)}")
    records: list[dict[str, object]] = []
    for row, (slug, subject, medium) in enumerate(scenes, start=1):
        prompt = (
            ANCHOR_PROMPT_TEXT
            if slug == "brutalist_courtyard"
            else _prompt(
                subject,
                medium,
                row,
            )
        )
        records.append(
            {
                "design_row": row,
                "slug": slug,
                "subject": subject,
                "medium": medium,
                "prompt": prompt,
            }
        )
    return tuple(records)


FAMILY_PROMPTS: Final[dict[str, tuple[dict[str, object], ...]]] = {
    family: _build_family(scenes) for family, scenes in FAMILY_SCENES.items()
}
