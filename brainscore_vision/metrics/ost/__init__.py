from brainscore_vision import metric_registry
from .metric import OSTCorrelation

metric_registry['ost'] = OSTCorrelation

BIBTEX = """@article{kubiliusschrimpf2019brain,
  title={Brain-like object recognition with high-performing shallow recurrent ANNs},
  author={Kubilius, Jonas and Schrimpf, Martin and Kar, Kohitij and Rajalingham, Rishi and Hong, Ha and Majaj, Najib and Issa, Elias and Bashivan, Pouya and Prescott-Roy, Jonathan and Schmidt, Kailyn and others},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}"""

BIBTEX_CHARACTERIZATION = """@article{kar2019evidence,
  title={Evidence that recurrent circuits are critical to the ventral stream’s execution of core object recognition behavior},
  author={Kar, Kohitij and Kubilius, Jonas and Schmidt, Kailyn and Issa, Elias B and DiCarlo, James J},
  journal={Nature neuroscience},
  volume={22},
  number={6},
  pages={974--983},
  year={2019},
  publisher={Nature Publishing Group US New York}
}"""
