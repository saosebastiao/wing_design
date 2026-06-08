from .airfoil import naca_00xx_coords, naca_00xx_thickness
from .wing import WingSpec, build_wing_solid, oml_section_polyline
from .wingsails import small_wingsail, medium_wingsail, large_wingsail

__all__ = [
    "WingSpec",
    "build_wing_solid",
    "oml_section_polyline",
    "naca_00xx_coords",
    "naca_00xx_thickness",
    "small_wingsail",
    "medium_wingsail",
    "large_wingsail",
]
