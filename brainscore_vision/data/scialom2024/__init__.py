from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = ""  # to appear in a future article

DATASETS = ['rgb', 'contours', 'phosphenes-12', 'phosphenes-16', 'phosphenes-21', 'phosphenes-27', 'phosphenes-35',
            'phosphenes-46', 'phosphenes-59', 'phosphenes-77', 'phosphenes-100', 'segments-12', 'segments-16',
            'segments-21', 'segments-27', 'segments-35', 'segments-46', 'segments-59', 'segments-77', 'segments-100',
            'phosphenes-all', 'segments-all']

data_assembly_sha1 = {
    'rgb': 'b696e29b685bdc0eb606b5d322014d8f176970f8',
    'contours': '29bee2e1072b3d122ba6bb2a832b3dacf150918f',
    'phosphenes-12': '6ce14a636d926d5201ecadea655babf248b1904b',
    'phosphenes-16': '9bedce4847a1ed2d83719bff7afe906b0e7c68f3',
    'phosphenes-21': 'b0dfba30f758b1575585604c0403f7be6a25a1a8',
    'phosphenes-27': '17241955b6948b125d323079bbbffc770dadcfee',
    'phosphenes-35': '3b911c510af77ab8572f0c5a608dd4e9ec2209ee',
    'phosphenes-46': 'a94194e9d9b6ab336ec78c250769b644dbec6fe0',
    'phosphenes-59': 'a86b7c8e146682ba2a0444b7a0de66e3c6512c07',
    'phosphenes-77': '3b89ab4511fbf849dbefc2db05ded31194adbcaa',
    'phosphenes-100': 'a2e112e30d5008bc90e30cf1416937233f0c619a',
    'segments-12': 'edc574bb8739aa2cf4bad3862e324758d861a139',
    'segments-16': 'af77a298f6a8e974adc21353357fe5e0a773291d',
    'segments-21': '08947367230078d8b89bec3f931f102b536d51a8',
    'segments-27': 'ebc8ebaf68713a78053c8fc52cfe83e186d75004',
    'segments-35': '25973c9c29375c977058c596b11f0486260ce70c',
    'segments-46': 'e35a8eb902abb09c0b72fe10e4800f78eb839327',
    'segments-59': 'f003c70519b467314881ef1aaacef3617b3e30fe',
    'segments-77': '1115d6e1cca9c24f2a4d96b74ed131364348663a',
    'segments-100': '39283150887a71590a705b646c51040510e76c1e',
    'phosphenes-all': '052570f8cd128b5422a87d7ea39c1e63e9571124',
    'segments-all': 'fcc1de0c1d54493602bee18ac90dd49842d35f84'
}

data_assembly_version_id = {
    'rgb': 'RqYj32F7uv4m.yXME6Fd2Eg8m4IYOCpR',
    'contours': 'gUrncxOSRAncOrPX.JXeUlYI09ypULCM',
    'phosphenes-12': 'N4kENT1qaWyUls8DuOheG.7HXA4s5Zaa',
    'phosphenes-16': 'RqrsVj6WhpEJpO3eeARBqK_gFKyqsQON',
    'phosphenes-21': 'rJRR.H0LQ_EFwBI94JFY6PSlKuJmqcbC',
    'phosphenes-27': 'VuYzKwmqxr.xI4iUlm7noeYOEummFR5v',
    'phosphenes-35': 'w7E5ogbuYCy4h6J3Dqgq1YvJcdsI7Pmo',
    'phosphenes-46': 'PoP3RnvVs4qZ8t0gpVksqAqG7WIjNne4',
    'phosphenes-59': 'jQpciBODH1GCwchVSGeschzmfVWKp8aP',
    'phosphenes-77': 't1tGokLUnmda5MMEwtqGuPvf9COHCyLw',
    'phosphenes-100': 'EZSXq7KzHiMiWBH_TSW0DFNMpsExTDh0',
    'segments-12': 'nMoHIN5x4q3Ce6lH9S1l14kbSS.09kvN',
    'segments-16': 'TaB28Lio1i40_LlRIseQ4sxO4MEGp3a8',
    'segments-21': 'JJYPqlFDn61jOAUJoEtbkn7mg9.Bx3TR',
    'segments-27': 'LsPeDwH_TVFGXZRhXZwSL54zG6IVgxBW',
    'segments-35': 'RHDbKHmn2dDdnDGfSxMjRzqKkFoFNpeR',
    'segments-46': 'tEc_j6eJlduriACI_OBY5flg04XFMS4m',
    'segments-59': 'MDqZ6mda76KmxXQRn5ZRYIF4zFOG2uXV',
    'segments-77': 'dz0D3kMCbmU2Snp0L3wp8laDyfljr7i1',
    'segments-100': '_vvxNL8hRm4svhgTkomTM5G1Bv3Np_iz',
    'phosphenes-all': 'J31X9HZJGuyi9srWEIaugmtNK9q8T2f7',
    'segments-all': 'Zyr7kaC_4j6F1LMKWAFCjlVphX5yvoAU'
}

stimulus_set_csv_sha1 = {
    'rgb': '405817a5dbc1bf9c655d5fcf76fddba075899873',
    'contours': '01e83b56500b94a18e536df8bcb90afe0983296f',
    'phosphenes-12': 'e9cc4e3a365c31bf3435599a56f42e87782696dc',
    'phosphenes-16': '16a22d3c3ab6bf645577b603d66dff469bbe7292',
    'phosphenes-21': '22d02f2af156d4f0c673f5a8be9a49b024058a8d',
    'phosphenes-27': 'ee2949bc86c85859aa7380fe6f8ff9962cf0f519',
    'phosphenes-35': 'dea437afcb5e78998d0e68c19d14fe15c317babb',
    'phosphenes-46': 'b12680c45ade05c4ac60ac9e0eb3a1eec0957540',
    'phosphenes-59': 'cbf182a88dc2c3323beae730f7dcae2f8d1258f3',
    'phosphenes-77': 'bf6affd6199c596565abbd04d051a89ffb41eda6',
    'phosphenes-100': 'e67fea553bc5eec32a9a8228d3f5599f296ea303',
    'segments-12': 'c75ec7b8097d10eb938dc3d37850dacd710b6c24',
    'segments-16': '976ada57727dbdfaffc99bea8727c9a3bc79eb50',
    'segments-21': 'ba4577bbe899d72deed9ee78af144c6c3802f8aa',
    'segments-27': '72189e59406cef27ca84230a020211178a458068',
    'segments-35': 'e2d41f2a44cad4bf8db00d6ef7b503ed0d75b90e',
    'segments-46': 'ed6af810d7db396f33166bc569b1ac669f472698',
    'segments-59': '073b3bdef04a3939644a89a93c6004c58a8cb75c',
    'segments-77': 'af32014049433fc7a54cbb57db1e1aa37ed1b812',
    'segments-100': 'bdb18259162739e0d933770a87c8eeb306dad061',
    'phosphenes-all': 'fc3593b3419ef61c0a96c4d27e98282e4c1e50eb',
    'segments-all': '6fdaf9fb31754995b373711b52664262dde2f54e'
}

stimulus_set_zip_sha1 = {
    'rgb': '6dce0513eb20c6d501a4e2e574028d12f910a1da',
    'contours': '70a2277a327b7aa150654f46f46ea4bd247c5273',
    'phosphenes-12': '701ecb4e2c87fac322a9d06b238e361d4c45b5bd',
    'phosphenes-16': '26563eff7a42561204fb6c158ae00ef4acf62e4d',
    'phosphenes-21': '22f9ecd333fcc177c16ed6c005989fe04f63700c',
    'phosphenes-27': '69f17f8b1121a19738625da78b4b73de2b151926',
    'phosphenes-35': 'e26e6e5fabca6bd52d85b38416df92de0b16b81e',
    'phosphenes-46': '6bb41a20bea70b3e521320258682dba1bef7a0e6',
    'phosphenes-59': '3e7d106796050a0fd4868567c8ef2325f294a13c',
    'phosphenes-77': '5b5ae2437193d75d26f5ffa87245ee479ce2674a',
    'phosphenes-100': 'b416d47663cd6e5d747f75f2f15eb52637313509',
    'segments-12': '0e04090ea7cc6cabc9efa7695dab5d217ebad70b',
    'segments-16': '2aa3cb8c89bf5956ea5c7a6c0f86ce8759056b41',
    'segments-21': 'e8116c9b1b2822bf2235b7ef91193baf2a7f47fb',
    'segments-27': '607d8589a270220046455e150d3589095d8270b1',
    'segments-35': '1df725d457f8092b2cc75a7f1509359383ae5420',
    'segments-46': '1f0bd6bcb476bff6937dc2bb0bf9222b381e41d5',
    'segments-59': '63cd9ee774ed4aeca54b8aa54c037037a67537d6',
    'segments-77': 'be2c2039e5eea7c3dcd90b885243fb6a1ebd40b9',
    'segments-100': '9d7ccb70f2231c8e62aecf18f1f48c160046c98b',
    'phosphenes-all': 'c2c26b17a4b4d152d1f20e78fe085e321080bcf6',
    'segments-all': '59b279b2c65be4cdab50bf17b7295da2426744ee'
}

stimulus_set_csv_version_id = {
    'rgb': 'tZjhmgUfU9FaYn01YwTz_yVWPgHRJ1bA',
    'contours': 'LUlDH2LLGUG2m797sYIDUKg6e8MdvFJk',
    'phosphenes-12': 'gDtgBj63aoJ1H.buKSJ.mA9GxW96Hqua',
    'phosphenes-16': 'IDnX9jbF1iiZ3A1zSg3nQC3c7fKnk_Qi',
    'phosphenes-21': '4ppMox5CbMDbBD44pyxppP0SNgdO4nqt',
    'phosphenes-27': 'QTceX4bryxXT1AvYWl2Guy6WE6sYPAmT',
    'phosphenes-35': '.cKTkhVwPZhhMU.nliMprH3AJCNWUJB_',
    'phosphenes-46': 'LCM9Hb9I_fPYKXbOZSD9DrsQbhW2QZWg',
    'phosphenes-59': 'Bg6z6733mjHYZdh.5oo5tjWabYMPN8BL',
    'phosphenes-77': 'oGb88.iLCTpLh7aOsqC3yi0.6wcRpBNC',
    'phosphenes-100': 'ZMZ7yM7axhU1D1ftuhKKO4dMS0.S4frU',
    'segments-12': 'v9kQqYE8Pu_eDz3BnmIYHk7f31bBYT0G',
    'segments-16': 'fmZsOPENCCAGOTHkfIhJSB423xuqwjle',
    'segments-21': 'k2WdSKbE9afwB0aQPgz.DMG8bLVOXb8m',
    'segments-27': 'XpAWpxju8bixupoVR7IfzS54Dvl25xB4',
    'segments-35': 'zgJMIxlciQU66oTPEZq0bVSohcqOHhZZ',
    'segments-46': '6v3Viy3GeeXPpkYP.hsceICCxj0GYXJt',
    'segments-59': 'CWUYHd7VKGJmFIbfBmMz73ao7.ITZqjY',
    'segments-77': 'XyQ7J..8HalBLX3ro7DxUYaSFgt1jc03',
    'segments-100': 'rffnTUErZ_AnzG6QqTYAtplcviMzQD4s',
    'phosphenes-all': 'V8y2NS9ERJzNgqRzBz5fcAOCjclWqfYK',
    'segments-all': 'eLnGaAJ1jXYRB9TtAR2KX1eKmmtWfJP0'
}

stimulus_set_zip_version_id = {
    'rgb': 'eYU9oxZr1YVWnP2oU6L.ExPgTtk7vVmV',
    'contours': 'SKXsJereWnqEbW8_JASu0ItMFR_czgtY',
    'phosphenes-12': 'smGooGiKZ5u5mQyhlfAr4.8FrCVmdWse',
    'phosphenes-16': '22wDqK.j4WqRcI2Hx7wLw7BJy84CmtB5',
    'phosphenes-21': 'vtvVcMD.fLftPKvzN._iqDdAAdOiDMCe',
    'phosphenes-27': 'uqlVmbQvTft6BbvCrm6r33XrmQnJekaZ',
    'phosphenes-35': 'O6RsIrqsciIaiY75GVcBKgKADeBuoG15',
    'phosphenes-46': 'kpjihqnS20Y8SqVA.0bYSu.nQOOJ4fxQ',
    'phosphenes-59': 'JhXSk.WFGAMJxsyXfQlXQvxbeR78axCd',
    'phosphenes-77': 'azseq3x19KMeAHo7FeB2.oVOLJgf7x45',
    'phosphenes-100': 'aRh9wHBprCh8G5OFJymKufBTaSYO.7u5',
    'segments-12': 'FKn9znL3g6q_q9adWpIMH0UhuKyu0l.t',
    'segments-16': '7YLj3ZpBl3m_pqjdRv66yg_F4lG2qTmb',
    'segments-21': 'KCYyGEQIN9_izF0v_neHFlcGV.HEZHn7',
    'segments-27': 'ICFDKnN1a47NRCUHDtlQZbnyJ_KxlaaE',
    'segments-35': 'i0RVls_jv1Z7o3x0_c.FH5yec4mGerU4',
    'segments-46': '4NIZLN18HH4pchhEhkxR5Hg5s84Yuipj',
    'segments-59': 'wfT5bVyAXduUN9jZIXLjpMj0cmsR052c',
    'segments-77': 'rBg1S.f5pDQaHoJGOvI5bpxkoyuGGZcA',
    'segments-100': 'KO3DVo1ZRJ9YbnaKjVE0MFy2WTOA3SYw',
    'phosphenes-all': 'SIJTyF92fwfNR_n9NgYnsbwEJFeKSsR5',
    'segments-all': 'DvY04p1.L77dZ4GXsf4.O7.ouSKjOXQn'
}



for dataset in DATASETS:
    # assembly
    data_registry[f'Scialom2024_{dataset}'] = lambda: load_assembly_from_s3(
        identifier=f'Scialom2024_{dataset}',
        version_id=data_assembly_version_id[dataset],
        sha1=data_assembly_sha1[dataset],
        bucket="brainio-brainscore",
        cls=BehavioralAssembly,
        stimulus_set_loader=lambda: load_stimulus_set(f'Scialom2024_{dataset}'),
    )

    # stimulus set
    stimulus_set_registry[f'Scialom2024_{dataset}'] = lambda: load_stimulus_set_from_s3(
        identifier=f'Scialom2024_{dataset}',
        bucket="brainio-brainscore",
        csv_sha1=stimulus_set_csv_sha1[dataset],
        zip_sha1=stimulus_set_zip_sha1[dataset],
        csv_version_id=stimulus_set_csv_version_id[dataset],
        zip_version_id=stimulus_set_zip_version_id[dataset])