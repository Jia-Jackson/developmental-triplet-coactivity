ZENODO MANUSCRIPT ARCHIVE -- VERSION 1.0.0

This record archives the data and code for:
Developmental triplet coactivity in mouse cortical cultures overlaps
lower-order and temporal structure.

Public repository: https://github.com/Jia-Jackson/developmental-triplet-coactivity
Original release: https://github.com/Jia-Jackson/developmental-triplet-coactivity/releases/tag/v1.0.0
Scientific commit: 170c82d4bf76a49eef5f42165c490305c6abd7a3

QUICK ACCESS
repository_upload.zip is a directly usable ZIP containing the exact 548-file
scientific repository snapshot: code, metadata, lossless spike-event exports,
processed measurements, statistical tables, and source data for all 19 figures.
Extract it and follow README.md. No volume reconstruction is needed for this ZIP.

LARGE ATTACHMENTS
Two raw-SPK ZIPs (DIV14 and DIV49) are supplied directly. The other six raw-SPK
ZIPs and analysis_inputs_and_checkpoints.zip are divided into ordinary 32-MiB
binary volumes named ORIGINAL.zip.part01, .part02, and so on. These are storage
volumes, not individual ZIP files. All original ZIP bytes are retained exactly.
The volumes avoid repeated failure of long single-request uploads; they do not
change any native SPK, model checkpoint, analytical result, or scientific code.

To restore the seven divided ZIPs:
1. Download all .partNN files, archive_manifest.json and restore_archives.py
   to the same folder.
2. Run: python restore_archives.py --folder "PATH_TO_DOWNLOADED_FILES"
3. The restored ZIPs appear in the restored_archives subfolder. The script
   checks every volume MD5 and each original SHA256 before accepting output.
   Existing differing files are never overwritten.
4. Extract the restored ZIPs, preserving their internal paths, as documented
   in the repository README.md.

archive_manifest.json maps every volume, byte offset and checksum to the
original public-release ZIP. Reassembly yields exactly the same bytes as the
unsplit ZIP assets downloadable from the public GitHub release.

CONTENTS AND LICENCES
Three independent wild-type mouse cortical preparations, 18 wells, eight ages
(DIV14,21,28,35,42,49,56,63), 143 retained recordings from 144 candidates.
WT_DIV35_P2_A1 is excluded; analysis-specific availability is documented.
24 native Axion SPK files contain detected spike events, not continuous voltage.
Research data and result tables: CC BY 4.0. Author-owned code: MIT.
Third-party maxent_toolbox retains its original MIT licence.
Final corrected prediction and endpoint tables take precedence over intermediate
tables as explained in the repository README.md. No new analyses were run for
archival packaging.
