# Publishing language packages

Publishing a GitHub release runs `.github/workflows/release.yml`. Rust and Node
publishers are independent of the existing Python and CLI release jobs.

Use a SemVer tag such as `v0.2.0` or `v0.2.0-rc.1`. Rust and npm versions are
set from the tag (without `v`); the Node lockfile is updated too. Build metadata
(`+...`) is rejected because npm strips it. npm prereleases use the `next`
dist-tag; stable releases use `latest`. A release marked as a GitHub prerelease
also uses `next`, even if its version has no prerelease suffix.

Rust ships portable CPU native sources and compiles on installation. Node ships
a small `clap-cpp` wrapper plus four separate optional packages:
`@gorse/clap-cpp-linux-x64`, `@gorse/clap-cpp-linux-arm64`,
`@gorse/clap-cpp-darwin-arm64`, and `@gorse/clap-cpp-win32-x64`.
The wrapper pins their versions exactly. npm uses each package's `os`/`cpu`
(and Linux `libc: glibc`) constraints to install only the matching runtime.
No clap.cpp install hook compiles or downloads binaries. The release builds
shared libraries externally, stages packages with `.github/scripts/pack-node.py`,
and tests actual npm selection against a temporary local registry on all four
platforms before publishing. All native packages publish before the wrapper.
The macOS runtime enables embedded Metal; Linux and Windows are CPU-only.

Rust source archives are tested on the same platform matrix. Cargo repackages
the extracted crate before upload: the publisher verifies that all files,
including the normalized manifest and native sources, match the tested archive.
Only Cargo-generated `Cargo.toml.orig` and VCS metadata are excluded; Rust
guarantees source equivalence, not a byte-identical compressed archive.

## Registry authorization

Configure GitHub environments `crates-io` and `npm`, including any required
reviewers. The workflow never creates registry credentials or configures trusted
publishers for you.

- **crates.io:** configure a GitHub trusted publisher for repository
  `gorse-io/clap.cpp`, workflow `release.yml`, environment `crates-io`.
  Alternatively, set environment/repository secret `CARGO_REGISTRY_TOKEN` with
  permission to publish `clap-cpp` (including first publication when needed).
  When this secret is set it is used instead of OIDC.
- **npm:** configure a GitHub trusted publisher for repository
  `gorse-io/clap.cpp`, workflow `release.yml`, environment `npm`.
  Alternatively, set environment/repository secret `NPM_TOKEN` to a granular
  publish token authorized for `clap-cpp` and all four `@gorse/clap-cpp-*`
  runtime packages, with the permissions/2FA bypass
  required for unattended publication. npm 11 is used for OIDC support;
  publications include provenance. Enable **direct publishing** for this trusted
  publisher because the workflow runs `npm publish`, not staged publication.
  A first publication may require a token
  before each package's trusted publisher can be configured. Configure the
  trusted publisher on all five packages, and ensure the publishing identity
  has permission to create/publish the scoped packages in the `@gorse` organization.

Check ownership and name availability before the first release. Neither registry
is contacted with a publish command by ordinary PR CI. Actual publication is
only exercised when a release is published and registry authorization is ready.

## Local checks

```sh
python -m unittest discover -s .github/scripts -v
python -m unittest discover -s rust/tests -p test_package.py -v
cargo package --allow-dirty -j 3
cmake --preset node-linux-x64
cmake --build build/node-runtime --config Release --target clap --parallel 4
cmake --install build/node-runtime --config Release --component runtime --prefix "$PWD/node/runtime/linux-x64"
cp LICENSE node/LICENSE
cp python/tests/tiny_models.py node/test/tiny_models.py
cd node
npm ci --omit=optional
python ../.github/scripts/pack-node.py --target linux-x64
mkdir -p artifacts
for package in dist/*; do npm pack "./$package" --pack-destination artifacts; done
npm install --no-save --package-lock=false --ignore-scripts ./artifacts/gorse-clap-cpp-*.tgz
npm run test:fixtures
npm test
npm run test:types
npm run test:packaging
python ../.github/scripts/test-node-install.py --artifacts artifacts --prefix installed
cd installed/node_modules/clap-cpp
npm run test:fixtures
npm test
```

The checkout must contain the pinned initialized submodules. The repository-root
Rust package includes original native sources and upstream licenses directly.
Node platform packages include only staged prebuilt runtimes and licenses, not
native sources or build scripts. The main package contains no shared libraries. Do not commit generated runtimes, fixture copies or
archives. See `node/README.md` for the other platform presets; release packaging
requires all four runtimes (omit `--target` when staging). Registry selection
tests use a local test registry with real tarballs, not a live npm publication.
CMake and a C++20 compiler are needed only for the separate native build; Python
is needed for the maintainer tests, not npm installation.
