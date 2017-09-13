import h5py
import getopt
import sys

def write_header(fout):
    fout.write("""
    <?xml version="1.0" ?>
    <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
    <Xdmf Version="2.0">
     <Domain>
        <Grid Name="mesh1" GridType="Uniform">
    """)

def write_geometry(fout, name, dims):
    fout.write("""
    <Topology TopologyType="3DSMesh" NumberOfElements="{1}"/>
     <Geometry GeometryType="X_Y_Z">
       <DataItem Dimensions="{1}" NumberType="Float" Precision="8" Format="HDF">
        {0}:/xmesh
       </DataItem>
       <DataItem Dimensions="{1}" NumberType="Float" Precision="8" Format="HDF">
        {0}:/ymesh
       </DataItem>
       <DataItem Dimensions="{1}" NumberType="Float" Precision="8" Format="HDF">
        {0}:/zmesh
       </DataItem>
     </Geometry>
    """.format(name, dims))

def write_attr(fout,name, dims, attr):
    fout.write("""
     <Attribute Name="{2}" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="{1}" NumberType="Float" Precision="8" Format="HDF">
        {0}:/{2}
       </DataItem>
     </Attribute>""".format(name, dims, attr))

def write_footer(fout):
    fout.write("""
       </Grid>
 </Domain>
</Xdmf>""")


def write_xmf(infile, outname):

    fout = file(outname, 'w')
    ds = h5py.File(infile)

    dims_arr = ds['O_p1_number_density'].shape
    dims = '{0} {1} {2}'.format(dims_arr[0], dims_arr[1], dims_arr[2])

    write_header(fout)
    write_geometry(fout, infile, dims)


    for var in ds.keys():
        if var in ['x', 'y', 'z', 'xmesh', 'ymesh', 'zmesh']: continue
        write_attr(fout,infile,  dims,  var)

    write_footer(fout)

    ds.close()
    fout.close()


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["infile="])
    except getopt.GetoptError:
        print getopt.GetoptError()
        return

    for opt, arg in opts:
        if opt == "--infile":
            infile = arg

    outname = infile.split('.')[0]+'.xmf'
    write_xmf(infile, outname)

if __name__ == "__main__":
    main(sys.argv[1:])
