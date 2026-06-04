@docs/guided_generative_design.md please analyze this document. I first want to fill in this document with necessary analysis and details, but then I
  want to create an incremental development plan to implement this idea. I will want to update the @docs/plan.md accordingly.

# Sailing Wing Design

A free-rotating, unstayed solid carbon-fiber **wingsail** built as a **space-frame of unidirectional CFRP beams** (straight or curved beams) wrapped and bound by a **filament-wound CFRP skin** that simultaneously forms the airfoil surface and spar, which is inserted into the boat hull similar to a keel-stepped mast, and rotates within its bearing-lined enclosure. The spar forms an effective pivot axis for 360+ degree rotation for the entire wing. 

## Glossary of Terms
TODO: infer or rename terms in this document to more clearly communicate to the AI agent as well as engineers and manufacturers. 
TODO: Define terms here for disambiguation

## Axis conventions
The wing is created in an x,y,z coordinate system
- the wing root airfoil is drawn on the x/y plane at z==0, and with the pivot point aligned at x==0 and y==0
- the wing tip airfoil is drawn on the x/y plan at z==`span`, also with the pivot point at (x,y)==(0,0)
- the wing spar is centered on (x,y)==(0,0)
    - the wing spar keel-step (the lowest point on the spar) is located at z== -(spar_length + spar_transition_length)
    - the wing spar deck-step (the highest point on the spar) is located at z== -spar_transition_length
- the wing chord line (whether at the root or tip or any intermediate z level) is always the x axis

## Wing Shape constraints specification 
- center of moment ahead of pivot axis
    - ensures that AoA passively decreases with increasing windspeed
- center of gravity should be aft of the pivot axis
    - ensures that AoA passively decreases when heeling
- wing spar centerpoint is also the pivot point
- wing spar radius is exactly the radius of the maximum inscribed circle at the wing spar centerpoint within the root aerofoil shape, rounded down to a whole centimeter
- there is a fillet between the spar and the spar transition
    - TODO: choose a reasonable default radius and name for this fillet
- there is a fillet between the spar transition and the wing root
    - TODO: choose a reasonable default radius and name for this fillet
- there is a fillet on the wing trailing edge that allows for continous filament winding of the wing shape
    - a pointed trailing edge would cause continous carbon fiber to bend in a way that weakens it, we need a minimized radius that still allows for the full strength of the fibers
    - TODO: choose a reasonable default radius and name for this fillet
- wing form beams always are along the outer shell of the wing 
    - this is necessary so that the filament winding of the shell creates the wing beam
    - this also doubles as the primary structural support for the wing
    - there needs to be enough of them to maintain a reasonable shape resolution for the wing, but some may carry more load than others

### Known Design Parameters                                         
span ( Wing Root to Wing Tip)  :            5.0 m                                         
wing_root_chord Root chord          : 1.0 m                                         
wing_root_airfoil       : NACA 0018 (symmetric, t/c = 0.18)             
wing_tip_chord          : 0.6 m                                         
wing_tip_airfoil        :  NACA 0018 (symmetric, t/c = 0.18)             
spar_transition_length : 0.5 m
spar_length : 1.0 m
pivot_location      : ~25% chord, move forward or backward to achieve predictable but small passive feathering moment forces. 
material (primary)  : UD carbon / epoxy (E1 ≈ 130 GPa, anisotropic) 
n_wing_form_beams: unknown
wing shell thickness: 3mm


## Aerodynamic Scenarios:
- full range of lift of foil (no need to exceed stall , as we are designing to passively feather)
- full range of reasonable sailing conditions
    - reynolds numbers, ncrit
- full range of reasonable wind shear
- full range of reasonable apparent winds
    - wind shear + boat speed == variable AoA along the wing
- beaufort force 11 winds at -1 to 1 degree AoA (our main failure criteria)

## structural simulation:
- structure must survive all scenarios within safety factor
- structure must not exceed tolerances for tip/twist/shape deflection
- stress/strain and principal directions of stresses at mesh cell or node level

## Performance Metrics and Constraints
### aerodynamic 
- moment force on pivot axis
- lift to drag ratio
- total lift
### structural 
- tip deflection
- twist deflection
- foil shape deflection

## Pipeline to achieve build design

1. build123d 
    - build wing outer solid for aerodynamic modeling
    - aerodynamic modeling only needs the wing section (from wing root to wing tip), not the spar or the spar transition
2. Use aerosandbox to generate pressure fields and point locations per load scenario (may need to decide on resolution)
3. 3D linear FEA - calculate Cauchy stress σ(x) 
4. Calculate stress-aligned frame field a la Arora
5. build123d build spar truss for wing
    1. build full wing (spar, transition, and wing) outer shell for structural modeling
    2. build collection of points for 3d splines for wing form beams 
        - all wing form beam splines are defined as points along the full wing outer shell
        - wing leading edge spline starts at the wing tip leading edge, proceeds to the wing root leading edge, then down the spar transition leading edge to the spar leading edge all the way down to the keel step
        - wing trailing edge spline does the same but with the trailing edge
        - we need multiple other splines that form a structural basis for maintaining the wing aerodynamic shape as well as the wings structural loads
            - let's start with a fixed number of evently spaced splines where each spline's x,y point is evenly spaced about the shell's filament-wound shell shape at whatever z level the point is at
            - choose the number of splines so that they roughly align with resolution of the aerodynamic pressure field vectors
        - choose a set of z axis points where we want spline points
    3. create the splines by connecting the points for each spar spline
        - optimize the spline knot vectors and weights such that they align as closely as possible with the wing shell shape
            - TODO: how do we do this?
6. solve LP model to optimize beam cross sections at each of the spline points that we've defined
    1. build a 3d polyline truss model from the chosen beam spline points that becomes the basis for a linear programming model which can be solved to optimize beam cross sections at each point along the polyline
        - assume unidirectional carbon fiber beams that form the shape of the polylines
        - each beam is modeled with fixed base at the spar keel-step
        - to model the stretched-skin structural effect of the wing shell being filament wound around the beams, form virtual beams between adjacent beams along the z-axis where the spline points are chosen. Assume infinite tension strength for these beams for this stage of the optimization. 
        - constrain the model to require tip/twist deflections below allowable values
        - optimization output: for each beam's spline point, we now should have an optimal cross section. 
    2. solve the model 
    3. iterate on the model if necessary to ensure that the model is producing coherent and realistic results. 
7. build123d create the beams 
    1. create x,y plane arcs with centers at all spline points and arc endpoints along the wing outer shell, facing inward from the shell
    2. at each beam spline point, solve for the arc radius in order to achieve the beam's specified cross sectional area that was output from the optimization model in step 6 above
    3. to form each beam, create lofts from the bottom of the spline to the top, aligning the arc points appropriately for the loft construction.
8. build123d create the wing wrap
    1. create horizontal lines from the beam arc termination points at each z-level to form a closed shape at each z-level
    2. loft the z-level closed shapes into a shell
    3. extrude the shell outwards by the shell thickness
9. build123d create assembly of beams and wing wrap
10. solve FEM model for structural capabilityh, generate metrics and visualizations



## future enhancements:
- generation / optimization of cross-beam members and reinforcements
- optimization of skin wrap weight vs beam weight

## Development Conventions:
1. Maintain a single dataclass definition which has all design parameters
    - only necessary parameters should be class members: if a parameter can be mathematically derived from another parameter, it shouldn't be maintained as a class member, but rather as a method on that class
2. Maintain an instance of that class with our default design parameters