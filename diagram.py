from diagrams import Diagram, Cluster, Node
from diagrams.generic import *
from diagrams.elastic.agent import Integrations
from diagrams.elastic.elasticsearch import SearchableSnapshots
from diagrams.elastic.beats import Packetbeat

with Diagram("SeqGAN Architecture", show=False):
    with Cluster(""):
        discriminator = SearchableSnapshots("Discriminator")
        rollout_policy = Packetbeat("Rollout Policy")
        generator = Integrations("Generator")

        rollout_policy >> generator
        discriminator << generator

    generator >> rollout_policy

diagram = Diagram("SeqGAN Architecture", show=False)