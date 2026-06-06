import numpy as np

from wing_design.geometry import WingSpec
from wing_design.materials.unidir import T700_EPOXY
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.laminate_sizing import LaminateSizingConfig, size_beam_shell_laminate


def test_clt_cosizing_feasible_and_valid_fractions():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    n = model.beam_elements.shape[0]
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 200.0
    loads[model.tip_nodes, 0] = 200.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e8, tip_defl_max_m=1.0, tip_twist_max_deg=2.0,
        r_min=0.004, r_max=0.03, t_min=0.0005, t_max=0.01,
    )
    res = size_beam_shell_laminate(model, [loads], cfg, ply=T700_EPOXY, rho=1550.0, maxiter=60)
    assert res.radii.shape == (n,)
    assert res.f0 >= -1e-6 and res.f45 >= -1e-6 and res.f90 >= -1e-6
    assert abs(res.f0 + res.f45 + res.f90 - 1.0) < 1e-6
    assert cfg.t_min - 1e-9 <= res.t_skin <= cfg.t_max + 1e-9
    assert res.max_beam_vm_Pa <= cfg.sigma_allow_Pa * 1.05
    assert res.max_skin_vm_Pa <= cfg.sigma_allow_Pa * 1.05
