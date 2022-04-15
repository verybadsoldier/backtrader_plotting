# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2022.04.15
### Fixes
- Fixed paramval2str type error for non-str lists


## [2.0.0] - 2021.06.14
### Added
- Live trading support (needs custom `backtrader`, see `README.md` for details)
- Data replaying support
- Customziable tab panels
- Added support for most marker types from https://matplotlib.org/3.2.1/api/markers_api.html
- Added line width support
- Added tag_pre_text_color to scheme. This fixes Tradimo theme where text color is background color for pre el.
- Automatically open browser after generation
- Made analyzer tables responsive in width
- Added support for indicator lines plotting areas (_fill_gt/_fill_lt)
- Improved labeling of plot entities
- Added support for MultiCoupler
- Added scheme parameter `strategysrc` to disable including stratgy source code in the meta data tab
- Allow `*` as index in `#`-plotconfig to target all plots of the type
- Lots of other things...


### Changed
- Disabled analyzer AnnualReturn for live mode

### Fixes
- Fixed crash on LineAction indicators (cannot be plotted though)
- Fixed xaxis not being rendered at all when using xaxis_pos="bottom" (#62)
- Fixed subplots activating plotmasters even though they were inactive
- Fixed xaxis_pos 'bottom' not working for tabs as bottom will only be decided after assigning tabs
- Also, various other fixes...

## [1.1.0] - xxxx.xx.xx
### Added
- stuff

## [1.0.0] - xxxx-xx-xx
### Changed
- stuff

## [0.5.3] - xxxx-xx-xx
### Changed
- stuff

[Unreleased]: https://github.com/verybadsoldier/backtrader_plotting/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/verybadsoldier/backtrader_plotting/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/verybadsoldier/backtrader_plotting/compare/v0.5.3...v1.0.0
[0.5.3]: https://github.com/verybadsoldier/backtrader_plotting/compare/v0.5...v0.5.3
